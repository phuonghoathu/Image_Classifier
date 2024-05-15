# Image_Classifier
In this project, I will building an image classifier and object detection system using PyTorch. 
# Dataset
Using CIFAR-10 dataset.  https://www.cs.toronto.edu/~kriz/cifar.html
![image](https://github.com/phuonghoathu/Image_Classifier/assets/5907880/b7b22d03-7f6d-4dfb-a04b-a31640da9a04)
# Information: 

Accuracy: 54.61

Epochs : 20

Model

    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(32*32*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
       # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # softmax or log_softmax
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
