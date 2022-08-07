import cv2, sys
import numpy as np
from mpi4py import MPI
import json
from random import randrange
import os
import math

# Determine if a number is a whole square root
def isSquare(number):
  x = number // 2
  seen = set([x])
  while x * x != number:
    x = (x + (number // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

# Load all images from the designated folder specified
def loadImagesFromFolder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Process the image, just GrayScaling it here
def processImage(image):
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
  return image

# The Convolution Algorithm
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

    # Command line arguements, first argument is the file, then gui, then kernal size, then kernal
    arg_names = ['file', 'gui', 'kernalSize', 'kernal']
    args = dict(zip(arg_names, sys.argv))

    # Get basic MPI communications, size and rank
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Is the kernel, but in 1 dimension
    kernelBuff = None

    # Get kernal size, defaults to 3x3
    kernalSize = args['kernalSize'] if "kernalSize" in args else '3x3'
    # Get GUI, default shows (showGUI arg). hideGUI in terminal arguments to not show it
    gui = args['gui'] if 'gui' in args else 'showGUI'
    # Kernal data, if random, randomize it. If passed in args parse it
    kernal = np.array(json.loads(args['kernal'])) if 'kernal' in args and args['kernal'] != 'random' else np.random.uniform(-1, 1, 25)

    if 'kernal' in args:
        kernelBuff = kernal
    else:
        # If no actuall data is given from args for kernal, default the first 9 values
        defaultBuff = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1], dtype=int)
        kernelBuff = np.concatenate((defaultBuff, np.zeros(16, dtype=int)), dtype=int)

    if comm_rank == 0 and gui == 'showGUI':
        # Load the GUI
        import PySimpleGUI as sg

        sg.theme('DarkAmber')

        # Default Kernal
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
        # Set default layout
        layout = [
                    [sg.Text('Select Kernel Size')],
                    [sg.Listbox(['3x3','4x4','5x5'], enable_events=True,pad=5, size=(8, 3), key='size', default_values='3x3')],
                    [sg.Text('Kernel')],
                    kernel,
                    [sg.Button('Start Convolution', key='start'), sg.Button('Cancel')] ]


        window = sg.Window('Convolution', layout)

        while True:
            try:
                event, values = window.read()
                # If window is closed break the loop
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    break
                # Retrieve kernal size
                kernalSize = values['size'][0]
                # Setting the layout if kernal size changes by user
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
                # Start the convolution, grab values and close window
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

    # Kernal is set via GUI or terminal, need to broadcast to other processes
    if ('kernal' in args and args['kernal'] == 'random') or gui == 'showGUI':
        comm.Bcast([kernelBuff, MPI.INT] , root=0)
        kernalSize = comm.bcast(kernalSize , root=0)

    # Init some vars
    kernelArray = []
    tempArray = []
    count = 0
    mod = None
    maxIndex = None
    finalData = None
    images = None
    numberOfImages = 0

    # Set some vars depending on size of kernel
    if kernalSize == '3x3':
        mod = 3
        maxIndex = 9

    elif kernalSize == '4x4':
        mod = 4
        maxIndex = 16

    else:
        mod = 5
        maxIndex = 25

    # Turn one dimensional array into the kernal size specified
    for v in kernelBuff:
        count = count + 1
        if count > maxIndex:
            break
        tempArray.append(v)
        if (count % mod == 0):
            kernelArray.append(tempArray)
            tempArray=[]

    # Final kernal used in the algorithms
    kernelArray = np.array(kernelArray)

    # Load images on process 0
    if comm_rank == 0:
        images = loadImagesFromFolder('./images')
        numberOfImages = len(images)
        # Display the kernal to user
        print('Kernal:')
        print(kernelArray)

    # Broadcast out the number of images, for iterative purposes later
    numberOfImages = comm.bcast(numberOfImages, 0)
    try:
        if comm_rank == 0:
            # Start time
            start = MPI.Wtime()
            # Work on one image at a time
            for image in images:
                # Grayscale Image
                image = processImage(image)
                # Buffer
                finalData = np.zeros((image.shape[0], image.shape[1]))
                # Breakup image and send to processes
                slices = np.vsplit(image, comm_size)
                # Send out all slices to the other processors
                for i in range(1, comm_size):
                    comm.send(slices[i], dest = i, tag = i)

                # The vertical stacked final array
                outList = []
                tempOutList = []
                for i in range(1, comm_size):
                    # Receive back the work from the other processors
                    tempOutList.append(comm.recv(source = i, tag = i))
                # Add process 0 to the vertical stacked final array first
                outList.append(convolve2D(slices[0], kernelArray, padding=0))
                # Then add the rest of the work done by the rest of the processors
                outList = outList + tempOutList
                # Stack it together for final output
                output = np.vstack(outList)
                # Write to output folder
                cv2.imwrite('./output/Standard-'+str(randrange(10000,2000000))+'.jpg', output)

            end = MPI.Wtime()
            print("Number of processors: {}, Standard algorithm seconds elapsed: {}".format(comm_size,end-start))
        else:
            for i in range(0, numberOfImages):
                # For each image receive the slice from processor 0
                received = comm.recv(source = 0, tag = comm_rank)
                # Do the work
                outputSegment = convolve2D(received, kernelArray, padding=0)
                # Then send it back to processor 0
                comm.send(outputSegment, dest = 0, tag = comm_rank)

    except Exception as e:
        print('Can\'t split image size evenly with the number of processors. The images are 512x512: ',e)

    # This algorithm only works if the number of processors can have a perfect square
    if isSquare(comm_size):
        # Algorithm 1 done, wait for all processors for Algorithm 2
        comm.Barrier()

        # Give all processors the images
        images = comm.bcast(images, root = 0)

        # For stats
        if comm_rank == 0:
            start = MPI.Wtime()


        for image in images:
            # Get the square root dimensions for cartology purposes
            sq_dim = int(math.sqrt(comm_size) if comm_size > 2 else comm_size)

            # Split the image into nprocessor slices vertically and nprocessor slices horizontally on processor 0
            slicesXY = []
            if comm_rank == 0:
                slicesX = np.vsplit(processImage(image), sq_dim )
                slicesXY = []
                for x in slicesX:
                    slicesXY.append(np.hsplit(x, sq_dim ))
                slicesXY = np.array(slicesXY)

            # Broadcast the data to all processors
            slicesXY = comm.bcast(slicesXY, root = 0)

            # Create the Cartesian topology, nprocessor = 16, cart is 4x4. nprocesssor = 4 cart is 2x2
            cartesian2d = comm.Create_cart(dims = [sq_dim, sq_dim] ,periods = None ,reorder=False)
            # Get assigned coordinates to each processor
            coord2d = cartesian2d.Get_coords(comm_rank)
            # Grab a slice of the image
            slice = slicesXY[coord2d[0]][coord2d[1]]
            # Run the 2d convolution on the slice
            outputSegment = convolve2D(slice, kernelArray, padding=0)

            if comm_rank == 0:
                # Create a buffer to eventually set the data
                data = np.zeros((sq_dim, sq_dim, outputSegment.shape[0], outputSegment.shape[1]), dtype=int)

                # Assign processor 0's work to 0,0 coords
                data[0, 0] = np.array(outputSegment)
                x = 0
                y = 0
                newImage = []
                yStack = []
                # Retreive the data that has been worked on
                for i in range(1, comm_size):
                    # Get the coords to the specific processor
                    x, y = cartesian2d.Get_coords(i)
                    # Depending on the coords, set the data back
                    data[x, y] = comm.recv(source = i, tag = i)

                # Have to unsplit the data, that was previously split to hand off to other processors
                for x in range(0,sq_dim):
                    tempArray = []
                    for y in range(0,sq_dim):
                        tempArray.append(data[x,y])
                    yStack.append(np.hstack(tempArray))
                # Have the image back intact with the convolution algorithm affecting it
                newImage = np.vstack(yStack)
                # Write out the image to output folder
                cv2.imwrite('./output/CartTop-'+str(randrange(10000,2000000))+'.jpg', newImage)

            else:
                # Send that slice to processor 0
                comm.send(outputSegment, dest = 0, tag = comm_rank)

        # Stats
        if comm_rank == 0:
            end = MPI.Wtime()
            print("Number of processors: {}, Cartesian topology algorithm seconds elapsed : {}".format(comm_size,end-start))
    else:
        if comm_rank == 0:
            print('Sorry, the number of processors have to be a perfect square. 4, 16, 25')