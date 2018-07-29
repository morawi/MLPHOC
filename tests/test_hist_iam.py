import cv2
import time
import matplotlib.pyplot as plt
# train data: 46945
# valid data: 6445
# test data: 13752

'''
Max Image Height is 241 n02-049-03-02 (182, 241) test set
Max Image Width is 1087 c06-103-00-01 (1087, 199) train set

'''
IMG_HEIGHT = 64
IMG_WIDTH = 1011 # m01-084-07-00 max_length
baseDir = 'datasets/IAM-V3/iam-images/' 
gg  = '/home/malrawi/Desktop/My Programs/MLPHOC/datasets/IAM-V3/iam-ground-truth/'
gt1 = gg+'RWTH.iam_word_gt_final.train.thresh'
gt2 = gg+'RWTH.iam_word_gt_final.valid.thresh'
gt3 = gg+'RWTH.iam_word_gt_final.test.thresh'

def calcLength(filename):
    global_length = []
    ss = 0
    data_gt_500w = {}
    h_vec = []
    w_vec=[]
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:#samples greater than 500: 425

                break
            word = line.split(' ')[0].split(',')[0] + '.png'
            img = cv2.imread(baseDir+ word, 0)
            h, w = img.shape            
            w_vec.append(w)
            h_vec.append(h)
            if w>800:
                ss +=1
                data_gt_500w.update({line.split()[0].split(',')[0]: (w,h)})
                
            global_length.append(w) # (IMG_HEIGHT/h*w)
    print("Avg W", sum(w_vec)/len(w_vec))
    print("Avg H", sum(h_vec)/len(h_vec))
    print('#samples greater than 500 is:', ss, end="; ")
    
    return global_length, data_gt_500w

s_t = time.time()
tr_length, data_gt_500w_trn = calcLength(gt1)
print('calc train length %.3fs' % (time.time() - s_t))
s_t = time.time()
va_length, data_gt_500w_val = calcLength(gt2)
print('calc valid length %.3fs' % (time.time() - s_t))
s_t = time.time()
te_length, data_gt_500w_tst = calcLength(gt3)
print('calc test length %.3fs' % (time.time() - s_t))

n, bins, patches = plt.hist(tr_length, 1000)
plt.title('IAM word length histogram-train')
plt.savefig('rwth_hist_train.png')
plt.gcf().clear()

n, bins, patches = plt.hist(va_length, 1000)
plt.title('IAM word length histogram-valid')
plt.savefig('rwth_hist_valid.png')
plt.gcf().clear()

n, bins, patches = plt.hist(te_length, 1000)
plt.title('IAM word length histogram-test')
plt.savefig('rwth_hist_test.png')
