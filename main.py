import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from PIL import Image, ImageOps, ImageGrab

# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Используемое устройство: {device}')

# Загрузка датасета MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Загрузка данных
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Определение модели
class Neural(nn.Module):
    def __init__(self):
        super(Neural, self).__init__()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.flat(x)
        out = torch.relu(self.linear1(out))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out

# Инициализация модели, лосс-функции и оптимизатора
model = Neural().to(device)  # Перенос модели на устройство (GPU или CPU)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Функция обучения модели
def train_model(model, train_loader, loss_fn, optimizer, epochs):
    model.train()
    loss_history = []  # Список для хранения потерь по эпохам
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Перенос данных на устройство
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    return loss_history

# Функция тестирования модели
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Перенос данных на устройство
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

# Задаём количество эпох
epochs = 10

# Обучение модели и получение истории потерь
loss_history = train_model(model, train_loader, loss_fn, optimizer, epochs)

# Тестирование модели
test_model(model, test_loader)

# Визуализация графика потерь
plt.figure(figsize=(10,5))
plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', color='b')
plt.title('График потерь во время обучения')
plt.xlabel('Эпоха')
plt.ylabel('Потери (Loss)')
plt.xticks(range(1, epochs+1))
plt.grid(True)
plt.show()

"""
def recognize_digit(image):
    # Изменение размера изображения на 28x28
    img = image.resize((28, 28)).convert('L')  # Конвертируем в градации серого
    img = ImageOps.invert(img)  # Инвертируем изображение, чтобы фон был чёрным, а цифра белой
    img = np.array(img) / 255.0  # Нормализуем пиксели
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        device)  # Добавляем размерности для batch и каналов

    # Применяем модель
    model.eval()
    with torch.no_grad():
        output = model(img)
        predicted = torch.argmax(output, 1).item()  # Получаем предсказанную цифру
    return predicted


# Создание интерфейса для рисования
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифры")
        self.root.geometry("400x400")

        self.canvas = Canvas(self.root, width=420, height=420, bg='white')
        self.canvas.pack()

        self.button_clear = Button(self.root, text="Очистить", command=self.clear_canvas)
        self.button_clear.pack(side='left')

        self.button_predict = Button(self.root, text="Распознать", command=self.predict_digit)
        self.button_predict.pack(side='right')

        self.canvas.bind("<B1-Motion>", self.paint)

        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=8, fill='black', capstyle=ROUND,
                                    smooth=TRUE)
        self.last_x, self.last_y = event.x, event.y

    def predict_digit(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Захват изображения с холста
        image = ImageGrab.grab((x, y, x1, y1))
        digit = recognize_digit(image)
        print(f"Распознанная цифра: {digit}")

        # Сброс координат
        self.last_x, self.last_y = None, None


# Запуск приложения
root = Tk()
app = PaintApp(root)
root.mainloop()

"""