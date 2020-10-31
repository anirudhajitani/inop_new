from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import glob

if len(sys.argv) < 3:
    print ("Provide folder name and algorithm")
    exit(1)
folder = sys.argv[1]
algo = sys.argv[2]
if algo != 'salmut':
    algo = algo + '_eval'
os.chdir(f"./{folder}/results")
#file_list = glob.glob('median_*_*_salmut_20.npy')
file_list = glob.glob(f'median_*_*_{algo}_20.npy')
print ("FILE LIST", file_list, algo)
file_list.sort()
#eval_res = np.array((1000,3), dtype=float)
counter = 0
for f in file_list:
    print ("FILE : ", f)
    tmp_arr = np.load(f)
    if counter == 0:
        eval_res = tmp_arr
        counter += 1
        continue
    eval_res = np.concatenate((eval_res, tmp_arr), axis=0)

print ("Array : ", eval_res, eval_res.shape)
#np.save('median_salmut_try_20.npy', eval_res)
np.save(f'median_final_{algo}_20.npy', eval_res)

addn_files = False

if addn_files == True:
    file_list_ov = glob.glob(f'overload_med_*_*_{algo}.npy')
    file_list_of = glob.glob(f'offload_med_*_*_{algo}.npy')
    print ("FILE LIST", file_list_ov, algo)
    print ("FILE LIST", file_list_of, algo)
    file_list_ov.sort()
    file_list_of.sort()
    #eval_res = np.array((1000,3), dtype=float)
    counter = 0
    for j in range(len(file_list_ov)):
        print ("FILE : ", file_list_ov[j], file_list_of[j])
        tmp_arr_ov = np.load(file_list_ov[j])
        tmp_arr_of = np.load(file_list_of[j])
        if counter == 0:
            eval_res_ov = tmp_arr_ov
            eval_res_of = tmp_arr_of
            counter += 1
            continue
        eval_res_ov = np.concatenate((eval_res, tmp_arr), axis=0)

    #print ("Array : ", eval_res_ov, eval_res_of)
    #np.save('median_salmut_try_20.npy', eval_res)
    np.save(f'overload_final_med_{algo}.npy', eval_res_ov)
    np.save(f'offload_final_med_{algo}.npy', eval_res_of)
