from tkinter import *
from time import sleep
from PIL import Image, ImageDraw, ImageFilter
from main import *
import cv2

class GUI:
    def __init__(self, n = 100):
        self.root = Tk()
        self.root.title("Number Identification Project!")
        self.root.geometry("400x500")
        self.n = n

        self.lastx = 0
        self.lasty = 0
        self.UseClassifier = "KNN"
        self.NNTrained = False

        # Acquiring the MNIST dataset using the function GetMnist
        data, self.labels = GetMnist(-1)
        
        # Creating a KNN classifier
        self.KnnClassifier = Knn(data, self.labels, Testing=False)
        
        # Creating a Neural Netwok classifier using Scikit Learn
        self.NNClassifier = MLPClassifier(max_iter=300, hidden_layer_sizes = (12, 12), activation = 'logistic')
        self.NNClassifier.n_outputs_ = 10
        
        f, nx, ny = data.shape
        self.data_reshaped = data.reshape((f, nx * ny))

        # Creating an empty canvas
        self.canv = Canvas(self.root, width=n, height=n, bg="white")
        self.canv.bind("<Button-1>", self.savePosn)
        self.canv.bind("<B1-Motion>", self.addLine)
        self.canv.pack(pady=40)

        self.but = Button(self.root, fg="white", bg="blue", text="Clean", height=2, width=10, command=self.clean)
        self.but.pack()
        
        self.but1 = Button(self.root, fg="blue", bg="green", text="Predict", height=2, width=10, command=self.predict)
        self.but1.pack(pady=5)


        self.but2 = Button(self.root, fg="white", bg="blue", text="Train NN", height=2, width=10, command=self.TrainNN)
        self.but2.pack(side=LEFT)

        self.but3 = Button(self.root, fg="white", bg="blue", text="Use KNN", height=2, width=10, command=self.UseKNN)
        self.but3.pack(side=RIGHT)

        self.but4 = Button(self.root, fg="white", bg="blue", text="Load NN", height=2, width=10)
        self.but4.bind('<Button-1>', self.LoadNN)
        self.but4.pack()


        self.lab = Label(self.root, text="Prediction: ")
        self.lab.pack(pady = 10)
        self.lab.config(font=("", 15))


        self.lab1 = Label(self.root, text="Using: KNN")
        self.lab1.pack(pady = 10)
        self.lab1.config(font=("", 15))

        # Creating an empty PIL image
        self.image = Image.new(("L"), (n, n), 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.root.mainloop()

    def savePosn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def addLine(self, event):
        self.canv.create_line((self.lastx, self.lasty, event.x, event.y))
        self.draw.line([self.lastx, self.lasty, event.x, event.y], "white")
        self.savePosn(event)
    
    def clean(self):
        self.canv.delete("all")
        self.image = Image.new(("RGB"), (self.n, self.n), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        self.image.save("Original.png", 'png')
        guess = self.makeAGuess()
        self.lab['text'] = "Prediction: " + str(guess) 

    def makeAGuess(self):
        img_arr = np.asarray(self.image.resize((28, 28)).convert('L')) / 255.
        new_image = np.zeros(img_arr.shape)  
        new_image = img_arr ** (1 / float(10))

        # new_image = cv2.imread()
        # new_image = cv2.erode(new_image, np.ones((5, 5), np.uint8), iterations=1)

        # cv2.imshow('Erosion', new_image)

        # plt.imshow(new_image)
        # plt.show()
        
        if self.UseClassifier == "KNN":
            return self.KnnClassifier.Predict(new_image)
        elif self.UseClassifier == "NN":
            return self.NNClassifier.predict(new_image.reshape(1, -1))

    def LoadNN(self, event):
        self.NNClassifier = joblib.load("NN.joblib")
        self.UseClassifier = "NN"
        self.setClassifier('NN')
        self.but2['text'] = "Use NN"
        self.NNTrained = True
        event.widget.pack_forget()


    def UseKNN(self):
        self.UseClassifier = "KNN"
        self.setClassifier('KNN')

    def TrainNN(self):
        if not(self.NNTrained):
            self.NNClassifier.fit(self.data_reshaped, self.labels)
            self.but2['text'] = "Use NN"
            self.NNTrained = True
        else:
            self.UseClassifier = "NN"
            self.setClassifier('NN')
    
    def setClassifier(self, t):
        self.lab1['text'] = "Using: " + t