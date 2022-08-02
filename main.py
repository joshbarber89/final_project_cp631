import cv2, sys
import numpy as np
import PySimpleGUI as sg

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
    padding = 0
    strides = 1
    # Edge Detection Kernel
    kernelArray = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
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

    # Create the Window
    window = sg.Window('Window Title', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
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
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break
        if event == 'start':
            try:
                kernelArray = []
                mod = 3
                if (kernalSize == '5x5'):
                    mod = 5
                elif kernalSize == '4x4':
                    mod = 4
                column = 0
                row = 0
                tempArray = []
                for k, v in values.items():
                    if type(k) == int:
                        tempArray.append(int(v))
                        if ((k + 1) % mod == 0 and k != 0):
                            kernelArray.append(tempArray)
                            tempArray=[]


                # Grayscale Image
                image = processImage(sys.argv[1])

                # Convolve and Save Output
                output = convolve2D(image, kernel, padding=2)
            except:
                print('Error')

    window.close()

