import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Modelin yüklenmesi
with open("model.p", "rb") as pickle_in:
    model = pickle.load(pickle_in)

# Görüntünün yüklenmesi
image_path = "traffic_Data/TEST/055_0080.png"
image = cv2.imread(image_path)

# Görüntünün boyutlarının ayarlanması ve gri tonlamaya dönüştürülmesi
height, width = 32, 32
resized_image = cv2.resize(image, (width, height))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Görüntünün normalleştirilmesi ve modele giriş için yeniden şekillendirilmesi
input_image = gray_image.reshape((1, height, width, 1)) / 255.0

# Model üzerinden tahmin yapılması
prediction = model.predict(input_image)
index = np.argmax(prediction)

# Sınıf adlarının tanımlanması
class_names = ['Hız sınırı (5 km/saat)',
               'Hız sınırı (15 km/saat)',
               'Hız sınırı (30 km/saat)',
               'Hız sınırı (40 km/saat)',
               'Hız sınırı (50 km/saat)',
               'Hız sınırı (60 km/saat)',
               'Hız sınırı (70 km/saat)',
               'Hız sınırı (80 km/saat)',
               'Düz veya sola gitmeyin',
               'Düz veya sağa gitmeyin',
               'Düz gitmeyin',
               'Sola dönülmez',
               'Sola veya sağa dönülmez',
               'Sağa dönülmez',
               'Soldan geçmeyin',
               'U dönüşü yapılmaz',
               'Araba ile girilmesi yasaktır',
               'Sesli ikaz cihazlarının kullanımı yasaktır',
               'Hız sınırı (40 km/saat)',
               'Hız sınırı (50 km/saat)',
               'İleri ve sağa mecburi yön',
               'İleri mecburi yön',
               'İlerde sola mecburi yön',
               'Sağa ve sola mecburi yön',
               'İlerde sağa mecburi yön',
               'Soldan gidiniz',
               'Sağdan gidiniz',
               'Ada etrafından dönünüz',
               'Mecburi motorlu taşıt yolu',
               'Korna çal',
               'Mecburi bisiklet yolu',
               'U dönüşü',
               'Her iki taraftan engeli geç',
               'Işıklı işaret cihazı',
               'Tehlikeli bölge',
               'Yaya geçidi',
               'Bisiklet geçebilir',
               'Okul geçidi',
               'Sola doğru tehlikeli viraj',
               'Sağa doğru tehlikeli viraj',
               'Yokuş aşağı uyarısı',
               'Dik yokuş uyarısı',
               'Sağa veya düz git',
               'Sola veya düz git',
               'Demiryolu geçidi',
               'Çalışma var',
               'Çit işareti',
               'Yol ver',
               'Duraklamak ve park etmek yasaktır',
               'Girilmez']

# Tahmin edilen sınıfın adının alınması
predicted_class = class_names[index]
print("Tahmin:", predicted_class)

# Görüntünün gösterilmesi
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(predicted_class)
plt.axis('off')
plt.show()
