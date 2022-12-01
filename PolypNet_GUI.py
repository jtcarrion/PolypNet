import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import LoadModel as lm
from PIL import Image, ImageTk

global IMAGE
IMAGE = ''

my_w = tk.Tk()

my_w.geometry("850x450")  # Size of the window
my_w.title('PolypNet')
my_font1=('times', 18, 'bold')
blank_img = "C:/Users/16023/Desktop/BMI_540_Project/blank.png"
ans_img = "C:/Users/16023/Desktop/BMI_540_Project/polyp.png"

def upload_file():
    global img
    f_types = [('png Files', '*.png')]
    filename = askopenfilename(filetypes=f_types)
    IMAGE = filename
    img= Image.open(filename)
    #img_resized= img.resize((350,250))
    img = ImageTk.PhotoImage(img)
    b4 = tk.Button(my_w, image=img)
    b4.grid(row=3, column=1, pady=(15))


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
    global img2
    global i
    i = lm.mask_up(IMAGE)
    img2 = ImageTk.PhotoImage(i)
    out = tk.Button(my_w, image=img2)
    out.grid(row=3, column=2)
    return()


def save_output():
    i.save("55_mask.png")
    tk.messagebox.showinfo(title=None, message='Image Saved!')
    return()


plc = Image.open(blank_img)  # loaded in white background
plc = plc.resize((300, 300))
new_plc = ImageTk.PhotoImage(plc)

l1 = tk.Label(my_w, text='Upload Colonoscopy Image', width=30, font=my_font1)
l1.grid(row=1, column=1)
l2 = tk.Label(my_w, text="Process Image", width=30, font=my_font1)
l2.grid(row=1, column=2)
# l3 = tk.Label(my_w, text="Save Image", width=30, font=my_font1)
# l3.grid(row=1,column=3)

b1 = tk.Button(my_w, text='Upload File', width=20, command=upload_file)
b1.grid(row=2, column=1, pady=(7))
b2 = tk.Button(my_w, text="Process Image", width=20, command=process_image)
b2.grid(row=2, column=2, pady=(7))
b3 = tk.Button(my_w, text="Save Image", width=20, command=save_output)
b3.grid(row=4, column=2, pady=(7))

label = Label(my_w, image=new_plc)  # "placeholders" for images
label.grid(row=3, column=1, pady=(7))
two_label = Label(my_w, image=new_plc)
two_label.grid(row=3, column=2, pady=(7))

my_w.mainloop()
