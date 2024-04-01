# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:41:17 2024

@author: Sam
"""

from pyrouge import Rouge155
import time

system_dir = r'C:\Users\Sam\Projects\CS584\HW2\System_Summaries'
system_filename_pattern = 'd(\d+)t.'
model_dir = r'C:\Users\Sam\Projects\CS584\HW2\Human_Summaries\eval' 
model_filename_pattern = 'D(\d+).M.100.T.[A-Z]'
result = {}

for temp in ['Centroid', 'DPP', 'ICSISumm', 'LexRank', 'Submodular']:    
    r = Rouge155()
    r.model_dir = model_dir
    r.model_filename_pattern = model_filename_pattern
    r.system_dir = system_dir + '\\' + temp    
    r.system_filename_pattern = system_filename_pattern + temp

    t = time.time()
    
    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    
    result[temp] = {
        '1' : [
            output_dict['rouge_1_precision']*100, 
            output_dict['rouge_1_recall']*100, 
            output_dict['rouge_1_f_score']*100
            ],
        '2': [
            output_dict['rouge_2_precision']*100, 
            output_dict['rouge_2_recall']*100, 
            output_dict['rouge_2_f_score']*100
            ],
        'L': [
            output_dict['rouge_l_precision']*100, 
            output_dict['rouge_l_recall']*100, 
            output_dict['rouge_l_f_score']*100
            ],
        'time' : time.time()-t
        }