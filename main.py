from tkinter import *
from tkinter import filedialog
from PIL import Image
from numpy import asarray
import torch



class Screen(Tk):
    def __init__(self):
        super().__init__()
        self.geometry('300x300')
        self.label_file_explorer = Label(self,
                                    text="File Explorer using Tkinter",
                                    width=100, height=4,
                                    fg="blue")
        self.label_file_explorer.pack()
        self.file_input = Button(self, text='file', command=self.browse_files)
        self.file_input.pack()
        self.button = Button(self, text='submit', command=self.submit_image)
        self.button.pack()
        self.filename = None
        self.file = None

    def browse_files(self):
        self.filename = filedialog.askopenfilename(initialdir = "",
                                              title = "Select a File",
                                              filetypes = (("Text files",
                                                            "*.png*"),
                                                           ("all files",
                                                            "*.*")))
        self.label_file_explorer.configure(text=self.filename)
        with open(self.filename) as file:
            self.file = file

    def submit_image(self):
        if self.filename:
            image = Image.open(self.filename)
            image = image.convert('L')
            image = image.resize((36, 36))
            image_data = asarray(image)
            tensor = torch.tensor(image_data)
            print(image_data)
            print(tensor)


screen = Screen()


screen.mainloop()