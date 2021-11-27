import numpy as np

def digitize(array: np.ndarray):
    '''
    Receives np.ndarray of shape (9,9); prints 
    formatted string to resemble sudoku game
    '''
    if array.shape != (9,9):
        raise Exception(f"Shape not supported; Provided {array.shape}, expected (9,9)")

    out = '\n'
    row_w = 37
    for i, line in enumerate(array):
        if i%3==0: 
            out += '='*row_w 
        # else:
        #     out += '-'*row_w
        out+='\n' 
        for j, num in enumerate(line):
            if j%3 == 0: 
                out += '|'
            else:
                out += ' '
            out += f' {str(num)} '
        out+='|\n'
    out += '='*row_w
    print(out)

if __name__ == '__main__':
    digitize(np.random.randint(0,10,81).reshape((9,9)))