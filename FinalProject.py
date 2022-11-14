import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk

my_w = tk.Tk()

my_w.geometry("800x500")  # Size of the window 
my_w.title('Final Project')
my_font1=('times', 18, 'bold')

l1 = tk.Label(my_w,text='Upload Colonoscopy Image',width=30,font=my_font1)  
l1.grid(row=1,column=1)
l2 = tk.Label(my_w, text="Process Image", width=30, font=my_font1)
l2.grid(row=1,column=2)
l3 = tk.Label(my_w, text="Save Image", width=30, font=my_font1)
l3.grid(row=1,column=3)

b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:open_file())
b1.grid(row=2,column=1) 
b2 = tk.Button(my_w, text="Process Image",width=20)
b2.grid(row=2,column=2)
b3 = tk.Button(my_w, text="Save Image",width=20)
b3.grid(row=2,column=3)

def upload_file():
    global img
    f_types = [('TIF Files', '*.TIF')]
    filename = askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    img_resized=img.resize((500,400))
    img = ImageTk.PhotoImage(img_resized)
    b4 =tk.Button(my_w,image=img) 
    b4.grid(row=3,column=1)

def open_file():
    global img
    f_types = [('TIF Files', '*.tif')]
    filename = askopenfilename(filetypes=f_types)
    new_img = Image.open(filename)
    new_img_resized = new_img.resize((500,400))
    new_img = ImageTk.PhotoImage(Image.open(new_img_resized))
    b4 =tk.Button(my_w,image=new_img) 
    b4.grid(row=3,column=1)

def process_image():
    return()

def save_output():
    filepath = asksaveasfilename

my_w.mainloop()