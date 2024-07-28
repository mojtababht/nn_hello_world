import torch
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
import random

from models import CNN
from training import device, train_save, test_data, loss_fn, optimizer, loaders


class Screen(Tk):
    def __init__(self, model):
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
        self.model = model
        self.last_output = None

    def browse_files(self):
        self.filename = filedialog.askopenfilename(initialdir = "",
                                              title = "Select a File",
                                              filetypes = (("Text files",
                                                            "*.png*"),
                                                           ("all files",
                                                            "*.*")))
        self.label_file_explorer.configure(text=self.filename)

    def submit_image(self):
        if self.filename:
            image = Image.open(self.filename)
            image = image.convert('L')
            image = image.resize((28, 28))
            image_arr = asarray(image, dtype=float)
            image_arr = image_arr / 256
            image_arr = asarray(image_arr, dtype='double')
            tensor = torch.tensor(image_arr, dtype=torch.float)
            tensor = tensor.unsqueeze(0)
            output = self.output(tensor)
            image = tensor.squeeze(0).squeeze(0).cpu().numpy()
            plt.imshow(image, cmap='gray')
            plt.show()
            is_true = messagebox.askquestion("Prediction", "Is prediction true?")
            if is_true == 'no':
                self.model.train()
                target = simpledialog.askstring(title="Test", prompt="What's real number:")
                while True:
                    try:
                        target = int(target)
                        break
                    except:
                        target = simpledialog.askstring(title="Test", prompt="enter a number:")
                loss = loss_fn(output, torch.tensor([target]).long().to(device))
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), 'data.pt')
                self.model.eval()
    def output(self, tensor):
        output = self.model(tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        print(f'prediction: {prediction}')
        return output



model = CNN().to(device)

try:
    model.load_state_dict(torch.load('data.pt'))
    raise Exception('mmd')
except:
    train_save()

model.eval()

# screen = Screen(model)
# screen.mainloop()


while True:
    model.eval()

    data, target = test_data[random.randint(0, 100)]
    for i, (x, y) in enumerate(loaders['train']):
        break

    data = data.unsqueeze(0).to(device)
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True).item()
    print(f'prediction: {prediction}, \t target: {target}')

    image = data.squeeze(0).squeeze(0).cpu().numpy()

    plt.imshow(image, cmap='gray')
    plt.show()
    # if input('exit(y or n):') == 'y':
    #     break