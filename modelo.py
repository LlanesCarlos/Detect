from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image


def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(24, 24),
              class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                   class_mode=class_mode, target_size=target_size)


BS = 32
TS = (24, 24)
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(SPE, VS)

# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # se usan 32 filtros, cada uno de tamaño 3x3
    # y otra vez
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),

    # se usan 64 filtros, cada uno de tamaño 3x3
    #  se eligen las mejores características por medio de pooling

    # se encienden y apagan neuronas aleatoriamente para mejorar convergencia
    Dropout(0.25),
    # se aplica Flatten pues hay muchas dimensiones, sólo se necesita una clasificación de salida
    Flatten(),
    # completamente conectado para obtener todos los datos relevantes
    Dense(128, activation='relu'),
    # se aplica un Dropout para mejorar la convergencia
    Dropout(0.5),
    # generar un 'softmax' para aplastar la matriz hacia probabilidades de salida
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

model.save('models/modelo.h5', overwrite=True)
