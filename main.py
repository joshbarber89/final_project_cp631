import cv2, sys
import numpy as np
import PySimpleGUI as sg
from mpi4py import MPI

def processImage(image):
  image = cv2.imread(image)
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
  return image


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    padding = None
    strides = None
    kernelBuff = np.empty(25, dtype=int)
    kernalSize = None

    if rank == 0:

        sg.theme('DarkAmber')

        kernel = [
                    [sg.InputText(-1, size=(2,1)),
                        sg.InputText(-1, size=(2,1)),
                        sg.InputText(-1, size=(2,1))],
                    [sg.InputText(-1, size=(2,1)),
                        sg.InputText(8, size=(2,1)),
                        sg.InputText(-1, size=(2,1))],
                    [sg.InputText(-1, size=(2,1)),
                        sg.InputText(-1, size=(2,1)),
                        sg.InputText(-1, size=(2,1))]
                    ]
        layout = [
                    [sg.Text('Padding')],
                    [sg.InputText(0, size=(2,1), key='padding')],
                    [sg.Text('Strides')],
                    [sg.InputText(1, size=(2,1), key='strides')],
                    [sg.Text('Select Kernel Size')],
                    [sg.Listbox(['3x3','4x4','5x5'], enable_events=True,pad=5, size=(8, 3), key='size', default_values='3x3')],
                    [sg.Text('Kernel')],
                    kernel,
                    [sg.Button('Start Convolution', key='start'), sg.Button('Cancel')] ]


        window = sg.Window('Convolution', layout)

        while True:
            try:
                event, values = window.read()

                if event == sg.WIN_CLOSED or event == 'Cancel':
                    break

                kernalSize = values['size'][0]
                padding = int(values['padding'])
                strides = int(values['strides'])
                if event == 'size':
                    if kernalSize == '3x3':
                        kernel = [
                            [sg.InputText(-1, size=(2,1)),
                                sg.InputText(-1, size=(2,1)),
                                sg.InputText(-1, size=(2,1))],
                            [sg.InputText(-1, size=(2,1)),
                                sg.InputText(8, size=(2,1)),
                                sg.InputText(-1, size=(2,1))],
                            [sg.InputText(-1, size=(2,1)),
                                sg.InputText(-1, size=(2,1)),
                                sg.InputText(-1, size=(2,1))]
                            ]
                    elif kernalSize == '4x4':
                        kernel = [
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))]
                            ]
                    elif kernalSize == '5x5':
                        kernel = [
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))],
                            [sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1)),
                                sg.InputText(0,size=(2,1))]
                            ]
                    layout = [
                        [sg.Text('Padding')],
                        [sg.InputText(0, size=(2,1), key='padding')],
                        [sg.Text('Strides')],
                        [sg.InputText(1, size=(2,1), key='strides')],
                        [sg.Text('Select Kernel Size')],
                        [sg.Listbox(['3x3','4x4','5x5'], enable_events=True, size=(8, 3), key='size', default_values=kernalSize)],
                        [sg.Text('Kernel')],
                        kernel,
                        [sg.Button('Start Convolution', key='start'), sg.Button('Cancel')] ]

                    newWindow = sg.Window('Window Title').Layout(layout)
                    window.Close()
                    window = newWindow
                    window.finalize()
                    window['size'].set_value(kernalSize)
                    window['padding'].update(padding)
                    window['strides'].update(strides)
                if event == 'start':
                    sendData = []
                    for k, v in values.items():
                        if type(k) == int:
                            sendData.append(int(v))
                    kernelBuff = np.array(sendData)
                    break
            except Exception as e:
                print('Error: ', e)
                break

        window.close()

    comm.Bcast([kernelBuff, MPI.INT] , root=0)
    kernalSize = comm.bcast(kernalSize , root=0)
    strides = comm.bcast(strides, root=0)
    padding = comm.bcast(padding, root=0)




    kernelArray = []
    tempArray = []
    count = 0

    if kernalSize == '3x3':
        mod = 3
        for v in kernelBuff:
            count = count + 1
            if count > 9:
                break
            tempArray.append(v)
            if (count % mod == 0):
                kernelArray.append(tempArray)
                tempArray=[]

    elif kernalSize == '4x4':
        mod = 4
        for v in kernelBuff:
            count = count + 1
            if count > 16:
                break
            tempArray.append(v)
            if (count % mod == 0):
                kernelArray.append(tempArray)
                tempArray=[]

    else:
        mod = 5
        for v in kernelBuff:
            count = count + 1
            tempArray.append(v)
            if (count % mod == 0):
                kernelArray.append(tempArray)
                tempArray=[]

    kernelArray = np.array(kernelArray)
    print(rank)
    print(kernelArray)
