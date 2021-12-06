import cv2 as cv
from matplotlib.pyplot import imshow, show, figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import keras
import os


def cv2_imshow(image, figsize=(3,3)):
    fig = figure(figsize=figsize)
    try:
        imshow(image[:,:,::-1])
    except IndexError:
        imshow(image, cmap="gray")
    show()

def extract_corners(filename):
    kernel = np.ones((3,3),np.uint8)
    #Read original image and grayscale
    img_original = cv.imread(filename)
    img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2GRAY)
    img_copy = img.copy()
    
    
    #Transform in black and white
    imgt = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,57,3)
    
    imgt_copy = imgt.copy()
    #Find external countour, create mask and draw external countour on mask
    contours, hierarchy = cv.findContours(image=imgt_copy, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(imgt)
    #cv2_imshow(imgt_copy)
    max_c = 0
    max_area = 0
    for contorno in contours:
        c_area = cv.contourArea(contorno)
        if c_area > max_area:
            max_area = c_area
            max_c = contorno
    cv.drawContours(image=mask, contours=[max_c], contourIdx=-1, color=(255), thickness=5, lineType=cv.LINE_4)
    #cv2_imshow(mask)
    corners = []
    for i in np.linspace(1, 0, 20): # Busca de melhor parâmetro
        #Find grid corners and draw on original image
        corners = cv.goodFeaturesToTrack(mask, 4, i, 50)
        if corners is None: continue
        for corner in corners:
            x,y = corner.ravel().astype(int)
        corners = sorted(corners.copy().astype(np.float32).reshape(corners.shape[0], corners.shape[2]), key=lambda x: x[0])
        if len(corners) == 4:
            break
    # Magical for-else loop, only enters else if finish loop without break
    else:
        raise Exception("Unable to find corners for supplied image! Are you sure it's a valid sudoku?")
    pts1 = []
    a,b,c,d = corners
    if a[1]>b[1]:
        pts1.append(list(a))
        pts1.append(list(b))
    else:
        pts1.append(list(b))
        pts1.append(list(a))
    if c[1]>d[1]:
        pts1.append(list(c))
        pts1.append(list(d))
    else:
        pts1.append(list(d))
        pts1.append(list(c))
    # Images for debugging
    # cv2_imshow(img_original)
    # cv2_imshow(imgt)
    # cv2_imshow(mask)
    # pts1 = Bot-Left; Top-Left; Bot-Right; Top-Right
    pts1 = np.array(pts1)
    return pts1

def produce_transform(filename: str, output_size: int):
    pts1 = extract_corners(filename)
    img_clean = cv.imread(filename)
    pts2 = np.float32([[0,output_size], [0,0], [output_size,output_size], [output_size,0]])
    # Draw circles on output, disabled for production, useful for debugging
    # for val in pts1:
    #     cv.circle(img_clean, [int(val[0]),int(val[1])], 1, (0,255,0), -1)
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img_clean,M,(output_size,output_size))
    #cv2_imshow(dst)
    return dst.copy()

def unsharp_mask(image, kernel_size=(7, 7), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def adaptive_threshold_cleanup(dst):
    clh = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    dst = clh.apply(cv.cvtColor(dst, cv.COLOR_BGR2GRAY))
    thresh = cv.threshold(dst, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 5))
    #thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    return thresh

def remove_borders(cell):
    img = np.zeros((28,28))
    img[3:27-3, 3:27-3] = cell[3:27-3, 3:27-3]
    return img.copy()

def show_all_cells(img):
    h, w = img.shape
    cell_size = int(h/9)
    min_x = 0
    min_y = 0
    for i in range(9):
        for j in range(9):
            cell_img = img[min_x:(min_x+cell_size), min_y:(min_y + cell_size)]
            resized = cv.resize(cell_img, (28,28), interpolation = cv.INTER_AREA)
            # cv2_imshow(resized)
            min_x += cell_size
        min_x = 0
        min_y += cell_size
        
def predict_all_cells(img, model):
    h, w = img.shape
    cell_size = int(h/9)
    min_x = 0
    min_y = 0
    out = []
    for i in range(9):
        for j in range(9):
            cell_img = img[min_y:(min_y + cell_size), min_x:(min_x+cell_size)]
            clean_img = remove_borders(cell_img)
            pred = model.predict(clean_img.reshape((-1,28,28,1)))
            max_value = max(pred[0])
            pred_index = np.where(pred[0] == max_value)
            # print(pred_index)
            out.append(pred_index[0])
            min_x += cell_size
        min_x = 0
        min_y += cell_size
    return out

def separate_cells(img):
    size = 28
    return img.reshape(img.shape[0]//size, size, -1, size).swapaxes(1,2).reshape(-1, size, size)/255.0

def analise_cell(cell):
    return int(cell[5:24, 5:24].sum() > 10)

def get_cell_array(img):
    t = separate_cells(img)
    return [analise_cell(i) for i in t]

def digitize(img, model):
    cell_arr = get_cell_array(img)
    preds = predict_all_cells(img, model)
    return np.array([c*p[0] for c,p in zip(cell_arr,preds)]).reshape((9,9))

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Greens, save_to_file = False):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize = (16,16))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_to_file:
        plt.savefig(title + '.pdf')
    return ax

def main(model_name: str, flag_test: int):
    FLAG_TEST = flag_test # 0 means testing CNN quality, 1 means testing Full model quality

    model = keras.models.load_model(model_name)
    log_str = f'{"="*40}\nSUMMARY\nModel = {model_name}\n{"="*40}\n\n'
    data_dir = 'sudoku/test_curated/jpg/'
    if FLAG_TEST:
        data_dir = 'sudoku/test_full/'
    total, total_hits = 0,0
    confusion_data = []
    total_imgs = len(list(filter(lambda x: x.endswith('.jpg'), os.listdir(data_dir))))
    errorlist = []
    for curr_img_index, img_fn in enumerate(list(filter(lambda x: x.endswith('.jpg'), os.listdir(data_dir)))):
        try:
            img_name = data_dir + img_fn
            with open(img_name.replace('jpg','dat'), 'r') as f:
                y = f.read()
                nums = y.split('\n')[2:]
                del nums[-1]
                nums = list(map(lambda x: x.replace(' ', ''), nums))
                arr_y = []
                for line in nums:
                    for char in line:
                        arr_y.append(int(char))
                arr_y = np.array(arr_y).reshape((9,9))
            result = adaptive_threshold_cleanup(produce_transform(img_name, 9*28))
            arr = digitize(result, model)
            bin_arr = get_cell_array(result)
            # print(f'{arr=}','\n', f'{arr_y=}')
            # print(img_name)
            # print(arr_y)
            # cv2_imshow(result)
            flat_arr_y = arr_y.ravel()
            flat_arr = arr.ravel()
            for i in range(len(flat_arr_y)):
                # if flat_arr_y[i] == 0: continue
                confusion_data.append([str(flat_arr_y[i]),flat_arr[i]])
            hits = (np.equal(arr, arr_y)*np.array(bin_arr).reshape((9,9))).astype(int).ravel().sum() 
            sub_total = sum(bin_arr)
            print(f'Score is {100*hits/sub_total:.02f}% or {hits}/{sub_total} ({curr_img_index+1}/{total_imgs})')
            log_str += f'Score for {img_fn} is {100*hits/sub_total:.02f}% or {hits}/{sub_total}\n'
            total += sub_total
            total_hits += hits
        except Exception as e:
            print(f'Encountered exception {e} at image {img_name}')
            log_str += f'Encountered exception {e} at image {img_name}\n'
            errorlist.append(img_name)
    log_str += f'\nTotal score is {100*total_hits/total:.02f}% or {total_hits}/{total}\n'
    log_num = len(list(filter(lambda x: x.endswith('log.txt'), os.listdir('./logs'))))
    log_name = f"./logs/{log_num}_log.txt" # Automatic log counter
    with open(log_name, 'w') as f:
        f.write(log_str)
    confusion_data = np.array(confusion_data)
    plot_confusion_matrix(confusion_data[:,0], confusion_data[:,1], list(map(lambda x: str(x), range(10))), save_to_file=True, title=f'./logs/experiment_{log_num}')
    print(f'Total score is {100*total_hits/total:.02f}% or {total_hits}/{total}')
    print(errorlist)

if __name__ == '__main__':
    for mn in ['CNN_MNIST_COMP_V1', 'CNN_MNIST_COMP_v2', 'CNN_MNIST_COMP_v3', 'CNN_MNIST_v1']:
        main(f'./{mn}', 0)
        main(f'./{mn}', 1)