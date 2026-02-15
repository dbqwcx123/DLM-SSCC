from __future__ import print_function
import argparse
import random
import os
import numpy as np
from torch.utils import data
import logging
import torch
import time
from pyldpc import make_ldpc
from utils.ECCT_utils import *

##################################################################

class My_Dataset(data.Dataset):
    def __init__(self, message, code, sigma):
        self.message = message
        self.code = code
        self.sigma = sigma
        self.generator_matrix = code.generator_matrix.transpose(0, 1)  # Matrix G, shape(k, n)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)  # Matrix transpose(H), shape(n, l)

    def __getitem__(self, index):
        # Check message length, split into groups of length code.k
        num_groups, message_list = split_message(self.message, self.code.k)
        m_list, x_list, z_list, y_list, magnitude_list, syndrome_list = [], [], [], [], [], []
        for i in range(num_groups):
            single_msg = message_list[i]
            
            # transmit each group of message into a tensor of shape (1, code.k)
            m = torch.tensor(single_msg).view(1, self.code.k)
            x = torch.matmul(m, self.generator_matrix) % 2  # encoded codeword
            z = torch.randn(self.code.n) * self.sigma[0]  # noise
            
            if channel == 'AWGN':
                h = 1
            elif channel == 'Rayleigh':
                h = torch.from_numpy(np.random.rayleigh(1, self.code.n)).float()
            else:
                raise ValueError("Invalid channel type.")
            
            y = h * bin_to_sign(x) + z  # received signal
            
            magnitude = torch.abs(y)
            syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                    self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            
            m_list.append(m.float())
            x_list.append(x.float())
            z_list.append(z.float())
            y_list.append(y.float())
            magnitude_list.append(magnitude.float())
            syndrome_list.append(syndrome.float())
            
        return m_list, x_list, z_list, y_list, magnitude_list, syndrome_list

    def __len__(self):
        return 1  # only 1 batch

##################################################################

def read_message_from_txt(filename):
    message_list = []
    with open(filename, 'r') as f:
        message_len, codeword_len = 0, 0
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
            message = [int(x) for x in message_string]
            message_list.append(message)
            message_len += len(message)
            codeword_len += (len(message)//code.k + 1) * code.n
    return message_list, message_len, codeword_len  # message

def split_message(message, k):
    """
    Splits the message list into sublists of length k. 
    If the last sublist is shorter than k, pads it with zeros. 
    Returns: 
        A list containing the sublists.
    """
    n = len(message)
    num_groups = (n + k - 1) // k
    result = []
    for i in range(num_groups):
        start = i * k
        end = min((i + 1) * k, n)
        group = message[start:end]
        if len(group) < k:
            group.extend([0] * (k - len(group)))
        result.append(group)
    return num_groups, result

##################################################################

def bitflip_decode(hard_decision, H, max_iter=100):
    """
    Bit-flip decoding algorithm
    hard_decision: initial hard decision bits (0/1)
    H: parity check matrix
    max_iter: maximum number of iterations
    """
    n = hard_decision.shape[0]
    m, n_h = H.shape
    
    # Ensure dimensions match
    assert n == n_h, f"Dimension mismatch: hard_decision has length {n}, H has {n_h} columns"
    
    # Convert to numpy for easier manipulation
    hd = hard_decision.clone().detach().numpy()
    H_np = H.clone().detach().numpy()
    
    # Iterative decoding
    for iteration in range(max_iter):
        # Calculate syndrome
        syndrome = np.dot(H_np, hd) % 2
        
        # Check if decoding is successful
        if np.sum(syndrome) == 0:
            break
        
        # Calculate number of unsatisfied parity checks for each bit
        unsatisfied = np.zeros(n)
        for j in range(n):
            for i in range(m):
                if H_np[i, j] == 1 and syndrome[i] == 1:
                    unsatisfied[j] += 1
        
        # Flip the bit with the maximum number of unsatisfied parity checks
        flip_idx = np.argmax(unsatisfied)
        hd[flip_idx] = 1 - hd[flip_idx]
    
    return torch.from_numpy(hd).float()

def estimate_bitflip(device, dataloader_list, SNR_range_test, code):
    test_loss_noise_ber, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    
    with torch.no_grad():
        for ii in range(len(dataloader_list)):  # num of [std_test] list
            test_loader = dataloader_list[ii]
            noise_ber = test_ber = test_fer = cum_count = 0.
            x_pred_list = []
            
            #########
            (m_list, x_list, z_list, y_list, magnitude_list, syndrome_list) = next(iter(test_loader))
            assert len(m_list) == len(x_list) == len(z_list) == len(y_list) == len(magnitude_list) == len(syndrome_list),\
                   "Length of lists in the test_dataloader inconsistent."
            
            # Get parity check matrix H (transpose of pc_matrix)
            H = code.pc_matrix.transpose(0, 1)
            
            for jj in range(len(m_list)):
                m, x, z, y, magnitude, syndrome = m_list[jj], x_list[jj], z_list[jj], y_list[jj], magnitude_list[jj], syndrome_list[jj]
                
                # Hard decision from received signal
                hard_decision = (y <= 0).float()
                noise_ber += BER(hard_decision, x)
                
                # Bit-flip decoding
                x_pred = bitflip_decode(hard_decision.squeeze(0), H.transpose(0, 1))
                x_pred = x_pred.unsqueeze(0)  # Add batch dimension back
                
                test_ber += BER(x_pred.to(device), x.to(device)) * x.shape[0]
                test_fer += FER(x_pred.to(device), x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                x_pred_list.append(x_pred.squeeze(0).cpu().numpy().astype(int))
            #########
            
            cum_samples_all.append(cum_count)
            test_loss_noise_ber.append(noise_ber / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            
            # Save bitflip predicted messages after channel transmission
            code_name = str(code.code_type)+'_K'+str(code.k)+'_N'+str(code.n)
            output_filename = f'./data/data/images/save_temp/cifar10/SNR_unified/{mode}/bitflip_pred/{code_name}/{channel}/demo_decode_SNR_{SNR_range_test[ii]}.txt'
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, 'a') as f:
                combined_x_pred = np.concatenate(x_pred_list)
                f.write("".join(map(str, combined_x_pred))+"\n")
                
        print('\nNoise BER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_noise_ber, SNR_range_test))]))
        print('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_fer_list, SNR_range_test))]))
        print('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_ber_list, SNR_range_test))]))
        print('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(snr, -np.log(elem)) for (elem, snr)
             in (zip(test_loss_ber_list, SNR_range_test))]))
    print(f'Test Time {time.time() - t} s\n')

    return test_loss_ber_list, test_loss_fer_list

##################################################################
class Code():
    pass

def main(args):
    code = args.code
    message_list, message_len, codeword_len = read_message_from_txt(args.msg_path)
    logging.info(f'Message total length: {message_len}')
    logging.info(f'Codeword total length: {codeword_len}')
    
    codeword_len_unified = 16*8*8*32*10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f'Using BitFlip decoding algorithm')
    logging.info(f'Transmission channel type: {args.channel}')
    
    #################################
    SNR_range_test = np.arange(-6, 16, 3)
    SNR_range_test_real = SNR_range_test + 10*np.log10(codeword_len_unified/codeword_len)
    logging.info(f'SNR_range_test: {SNR_range_test_real}')
    
    std_test = [SNR_to_std(ii) for ii in SNR_range_test_real]
    print(f'std_test: {std_test}')
    
    test_loss_ber_sum, test_loss_fer_sum = [0] * len(std_test), [0] * len(std_test)
    for message in message_list:
        dataloader_list = [My_Dataset(message, code, sigma=[std_test[ii]]) for ii in range(len(std_test))]
        
        test_loss_ber_list, test_loss_fer_list = estimate_bitflip(device, dataloader_list, SNR_range_test, code)
        
        test_loss_ber_sum = list(map(lambda x, y: x + y, test_loss_ber_list, test_loss_ber_sum))
        test_loss_fer_sum = list(map(lambda x, y: x + y, test_loss_fer_list, test_loss_fer_sum))
    
    test_loss_ber_avg = list(map(lambda x: x / len(message_list), test_loss_ber_sum))
    test_loss_fer_avg = list(map(lambda x: x / len(message_list), test_loss_fer_sum))
    
    code_name = str(code.code_type)+'_K'+str(code.k)+'_N'+str(code.n)
    metric_filename = f'./data/data/images/save_temp/cifar10/SNR_unified/{mode}/bitflip_pred/{code_name}/{channel}/#demo_decode_metric.txt'
    with open(metric_filename, "a") as f:
        f.write("\nBER:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_ber_avg]))
        f.write("\nFER:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_fer_avg]))

##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BitFlip Decoding')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0", help='gpus ids')
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Message args
    parser.add_argument('--msg_filename', type=str, default='compress_output')

    # Code args
    parser.add_argument('--code_type', type=str, default='LDPC',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=24)
    parser.add_argument('--code_n', type=int, default=49)
    parser.add_argument('--channel', type=str, default='Rayleigh',
                        choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--mode', type=str, default='igpt-l-modified')

    # BitFlip args
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum iterations for bitflip decoding')
    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    d_c = int(np.sqrt(code.n))
    d_v = int(d_c + 1 - code.k/(d_c-1))
    H, G = make_ldpc(code.n, d_v, d_c, systematic=True, sparse=True)
    code.generator_matrix = torch.from_numpy(G).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    channel = args.channel
    mode = args.mode
    
    ####################################################################
    
    msg_filename = args.msg_filename
    msg_dir = f'./data/data/images/save_temp/cifar10/SNR_unified/{mode}/compressed/{args.msg_filename}.txt'
    if not os.path.exists(msg_dir):
        raise FileNotFoundError(f"Message_dir {msg_dir} does not exist.")
    else:
        args.msg_path = msg_dir
    
    # Create output directory for bitflip results
    output_dir = f'./data/data/images/save_temp/cifar10/SNR_unified/{mode}/bitflip_pred/{code.code_type}_K{code.k}_N{code.n}/{channel}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_dir = f'./data/model/Results_BitFlip/{args.code_type}__Code_n_{args.code_n}_k_{args.code_k}_{args.channel}'
    os.makedirs(log_dir, exist_ok=True)
    
    handlers = [
        logging.FileHandler(os.path.join(log_dir, 'logging_test.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    print(f"Path to logs: {log_dir}")
    print(args)
    
    main(args)