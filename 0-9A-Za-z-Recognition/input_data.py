import os
from PIL import Image, ImageFilter

def getImagesWithLabels():
    dataset = []

    for dirname, dirnames, filenames in os.walk("Dataset"):
        for subdirname in dirnames:
            for dirname1, dirnames1, filenames1 in os.walk(os.path.join("Dataset", subdirname)):
                for filename1 in filenames1:

                    labelName = subdirname.split("_")[0]
                    indexIntoTensor = int(subdirname.split("_")[1])
                    fullPath = dirname + "\\" + subdirname + "\\" + filename1

                    im = Image.open(fullPath)
                    im1 = im.convert('RGB')
                    im2 = im1.resize((100, 100))
                    imbytes = list(im2.getdata())
                    imbytesOneRow = []
                    labelArr = [0] * 62
                    labelArr[indexIntoTensor] = 1

                    for pixel in imbytes:
                        imbytesOneRow.append(pixel[0])
                        imbytesOneRow.append(pixel[1])
                        imbytesOneRow.append(pixel[2])

                    dataset.append([fullPath, labelName, indexIntoTensor, imbytesOneRow, labelArr])


    return dataset