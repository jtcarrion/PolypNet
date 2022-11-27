import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import LoadModel as lm
from PIL import Image, ImageTk

my_w = tk.Tk()

my_w.geometry("800x500")  # Size of the window
my_w.title('Final Project')
my_font1=('times', 18, 'bold')
IMAGE = ""
blank_img = "C:/Users/16023/Desktop/BMI_540_Project/PolypNet-main/blank.png"
ans_img = "C:/Users/16023/Desktop/BMI_540_Project/polyp.png"

def upload_file():
    global img
    f_types = [('png Files', '*.png')]
    filename = askopenfilename(filetypes=f_types)
    IMAGE = filename
    img= Image.open(filename)
    img_resized= img.resize((350,250))
    img = ImageTk.PhotoImage(img_resized)
    b4 = tk.Button(my_w,image=img)
    b4.grid(row=3,column=1)


'''
def open_file():
    global img
    f_types = [('TIFF Files', '*.tif')]
    filename = askopenfilename(filetypes=f_types)
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    array = []
    for x in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(x)
        array = band.ReadAsArray()
    b4 = tk.Button(my_w, image=array)
    b4.grid(row=3, column=1)
'''

def process_image():
    file = IMAGE
    i = lm.mask_up(file)
    img = ImageTk.PhotoImage(i)
    out = tk.Button(my_w, image=img)
    out.grid(row=3, column=2)
    return()


def save_output():
    filepath = asksaveasfilename
    return()


plc = Image.open(blank_img)  # loaded in white background
plc = plc.resize((288, 368))
new_plc = ImageTk.PhotoImage(plc)

l1 = tk.Label(my_w, text='Upload Colonoscopy Image', width=30, font=my_font1)
l1.grid(row=1,column=1)
l2 = tk.Label(my_w, text="Process Image", width=30, font=my_font1)
l2.grid(row=1,column=2)
# l3 = tk.Label(my_w, text="Save Image", width=30, font=my_font1)
# l3.grid(row=1,column=3)

b1 = tk.Button(my_w, text='Upload File', width=20, command=upload_file)
b1.grid(row=2,column=1)
b2 = tk.Button(my_w, text="Process Image", width=20, command=process_image)
b2.grid(row=2,column=2)
b3 = tk.Button(my_w, text="Save Image", width=20)
b3.grid(row=4,column=2)

label = Label(my_w, image = new_plc) # "placeholders" for images 
label.grid(row=3,column=1)
two_label = Label(my_w, image= new_plc)
two_label.grid(row=3,column=2)

print(IMAGE)

my_w.mainloop()
