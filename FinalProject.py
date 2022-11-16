import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename

from PIL import Image, ImageTk
#from osgeo import gdal

my_w = tk.Tk()

my_w.geometry("800x500")  # Size of the window
my_w.title('Final Project')
my_font1=('times', 18, 'bold')

def upload_file():
    global img
    f_types = [('png Files', '*.png')]
    filename = askopenfilename(filetypes=f_types)
    img= Image.open(filename)
    img_resized= img.resize((500,400))
    img = ImageTk.PhotoImage(img_resized)
    b4 = tk.Button(my_w,image=img)
    b4.grid(row=3,column=1)

def open_file():
    global img
    f_types = [('TIFF Files', '*.tif')]
    filename = askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img = img.resize((300, 250))
    img = ImageTk.PhotoImage(img)
    my_image_label = tk.Label(my_w, image=img)
    my_image_label.grid(row=3, column=1)

def process_image():
    return()

def save_output():
    filepath = asksaveasfilename
    return()

l1 = tk.Label(my_w, text='Upload Colonoscopy Image', width=30, font=my_font1)
l1.grid(row=1,column=1)
l2 = tk.Label(my_w, text="Process Image", width=30, font=my_font1)
l2.grid(row=1,column=2)
#l3 = tk.Label(my_w, text="Save Image", width=30, font=my_font1)
#l3.grid(row=1,column=3)

b1 = tk.Button(my_w, text='Upload File', width=20, command=open_file)
b1.grid(row=2,column=1)
b2 = tk.Button(my_w, text="Process Image", width=20)
b2.grid(row=2,column=2)
b3 = tk.Button(my_w, text="Save Image", width=20)
b3.grid(row=3,column=2)

my_w.mainloop()
