import os

def get_avail_gpu():
    '''
    works for linux
    '''
    result = os.popen("nvidia-smi").readlines()

    try:
    # get Processes Line
        for i in range(len(result)):
            if 'Processes' in result[i]:
                process_idx = i

        # get # of gpus
        num_gpu = 0
        for i in range(process_idx+1):
            if 'MiB' in result[i]:
                num_gpu += 1
        gpu_list = list(range(num_gpu))

        # dedect which one is busy
        for i in range(process_idx, len(result)):
            if result[i][22] == 'C':
                gpu_list.remove(int(result[i][5]))
                
        return (gpu_list[0])
    except:
        print('no gpu available, return 0')
        return 0

