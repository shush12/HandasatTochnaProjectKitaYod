from tkinter import *
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
        self.NNClassifier = MLPClassifier(max_iter=200, hidden_layer_sizes = (150, 150), activation = 'logistic')
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
        self.image.save("Number.png", 'png')
        guess = self.makeAGuess()
        self.lab['text'] = "Prediction: " + str(guess) 

    def makeAGuess(self):
        # קורא את התמונה שנכתבה בפעולה הקודמת
        img = cv2.imread('Number.png')
        
        # מעבה את הפיקסלים הלבנים שבתמונה
        img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
        
        # משנה את הגודל של התמונה ל28 על 28 פיקסלים
        img = cv2.resize(img, (28, 28))
        
        # ממיר את התמונה לתמונה בשחור לבן בלבד
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # מנרמל את ערכי התמונה
        fin = img.reshape(1, -1) / 255.

        # הופך את הפיקסלים לבהירים יותר
        fin = fin ** (1 / float(7))

        # מכניס את התמונות לפעולת החיזוי
        if self.UseClassifier == "KNN":
            return self.KnnClassifier.Predict(fin)
        elif self.UseClassifier == "NN":
            return self.NNClassifier.predict(fin)

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
            self.but4.pack_forget()
        else:
            self.UseClassifier = "NN"
            self.setClassifier('NN')
    
    def setClassifier(self, t):
        self.lab1['text'] = "Using: " + t