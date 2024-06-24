from tkinter import *
from tkinter import filedialog


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
        self.button = Button(self, text='submit', command=lambda: print('hi'))
        self.button.pack()
        self.filename = None
        self.file = None

    def browse_files(self):
        self.filename = filedialog.askopenfilename(initialdir = "",
                                              title = "Select a File",
                                              filetypes = (("Text files",
                                                            "*.txt*"),
                                                           ("all files",
                                                            "*.*")))
        self.label_file_explorer.configure(text=self.filename)
        with open(self.filename) as file:
            self.file = file


screen = Screen()


screen.mainloop()