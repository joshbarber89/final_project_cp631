import cv2, sys
import numpy as np
from mpi4py import MPI
import json
from random import randrange
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def processImage(image):
  #image = cv2.imread(image)
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
  return image

def convolve2D(image, kernel, padding=0, strides=1):
    # Kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Image and Kernel Shapes
    inputImageShapeY, inputImageShapeX = image.shape[1], image.shape[0]
    kernelShapeY, kernelShapeX = kernel.shape[1], kernel.shape[0]

    # Output Convolution 2D
    output = np.zeros((int(((inputImageShapeX - kernelShapeX + 2 * padding) / strides) + 1), int(((inputImageShapeY - kernelShapeY + 2 * padding) / strides) + 1)))

    # Apply Equal Padding to All Sides
    if padding == 0:
        paddedImage = image
    else:
        paddedImage = np.zeros((padding*2 + image.shape[0], padding*2 + image.shape[1]))
        paddedImage[int(padding):int(padding * -1), int(padding):int(padding * -1)] = image

    # Iterate through whole image
    for y in range(image.shape[1]):
        # Exit Convolution
        if image.shape[1] - kernelShapeY < y :
            break
        # Only Convolve if y has gone down by the specified Strides
        if 0 == y % strides:
            for x in range(image.shape[0]):
                # Once kernel is out of bounds go to next row
                if image.shape[0] - kernelShapeX < x:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if 0 == x % strides:
                        output[x, y] = (paddedImage[x: x + kernelShapeX, y: y + kernelShapeY] * kernel).sum()
                except:
                    break

    return output

if __name__ == '__main__':

    # gui -> hideGUI / showGUI (default)
    # kernalSize -> 3x3 (default) / 4x4 / 5x5
    # kernal -> random / [x,x,x,x,x,x,x,x,x] / [-1,-1,-1,-1,8,-1,-1,-1,-1] (default)

    arg_names = ['file', 'gui', 'kernalSize', 'kernal']
    args = dict(zip(arg_names, sys.argv))

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    padding = None
    strides = None
    kernelBuff = None

    kernalSize = args['kernalSize'] if "kernalSize" in args else '3x3'
    gui = args['gui'] if 'gui' in args else 'showGUI'
    kernal = np.array(json.loads(args['kernal'])) if 'kernal' in args and args['kernal'] != 'random' else np.random.uniform(-1, 1, 25)

    if 'kernal' in args:
        kernelBuff = kernal
    else:
        defaultBuff = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1], dtype=int)
        kernelBuff = np.concatenate((defaultBuff, np.zeros(16, dtype=int)), dtype=int)

    if comm_rank == 0 and gui == 'showGUI':
        import PySimpleGUI as sg

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

    if ('kernal' in args and args['kernal'] == 'random') or gui == 'showGUI':
        comm.Bcast([kernelBuff, MPI.INT] , root=0)
        kernalSize = comm.bcast(kernalSize , root=0)
        strides = comm.bcast(strides, root=0)
        padding = comm.bcast(padding, root=0)

    kernelArray = []
    tempArray = []
    count = 0
    mod = None
    maxIndex = None
    if kernalSize == '3x3':
        mod = 3
        maxIndex = 9

    elif kernalSize == '4x4':
        mod = 4
        maxIndex = 16

    else:
        mod = 5
        maxIndex = 25

    for v in kernelBuff:
        count = count + 1
        if count > maxIndex:
            break
        tempArray.append(v)
        if (count % mod == 0):
            kernelArray.append(tempArray)
            tempArray=[]

    kernelArray = np.array(kernelArray)
    finalData = None
    numberOfImage = 0

    if comm_rank == 0:
        images = load_images_from_folder('./images')
        numberOfImage = len(images)


    numberOfImage = comm.bcast(numberOfImage, 0)


    start = MPI.Wtime()
    if comm_rank == 0:
        print('Kernal:')
        print(kernelArray)
        for image in images:
            # Grayscale Image
            image = processImage(image)
            finalData = np.zeros((image.shape[0], image.shape[1]))
            # Breakup image and send to processes
            slices = np.vsplit(image, comm_size)
            for i in range(1, comm_size):
                comm.send(slices[i], dest = i, tag = i)


            outList = []
            tempOutList = []
            for i in range(1, comm_size):
                tempOutList.append(comm.recv(source = i, tag = i))
            outList.append(convolve2D(slices[0], kernelArray, padding=0))
            outList = outList + tempOutList
            output = np.vstack(outList)
            cv2.imwrite('./output/2DConvolved-'+str(randrange(10000,2000000))+'.jpg', output)

        end = MPI.Wtime()
        print("Seconds elapsed: {}".format(end-start))
    else:
        for i in range(0, numberOfImage):
            received = comm.recv(source = 0, tag = comm_rank)
            outputSegment = convolve2D(received, kernelArray, padding=0)
            comm.send(outputSegment, dest = 0, tag = comm_rank)

