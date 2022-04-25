import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data = pd.read_csv('data_example.txt')

    data_shuffled = data.sample(n=len(data))

    x = data_shuffled[['delta_x', 'delta_y', 'x_vel', 'y_vel']]
    y = data_shuffled['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5)

    x_train = tf.convert_to_tensor(x_train)
    x_val = tf.convert_to_tensor(x_val)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_val = tf.convert_to_tensor(y_val)
    y_test = tf.convert_to_tensor(y_test)

    print("train")
    print(x_train)
    print(y_train)
    print("test")
    print(x_test)
    print(y_test)
    print("val")
    print(x_val)
    print(y_val)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[es])

    predictions = model.predict(x_test)

    print(predictions)
    print(y_test)


    def transform(predictions):
        result = []
        for prediction in predictions:
            if prediction[0] > prediction[1] and prediction[0] > prediction[2]:
                result.append(0)
            elif prediction[1] > prediction[2]:
                result.append(1)
            else:
                result.append(2)
        return result


    matrix = tf.math.confusion_matrix(y_test, transform(predictions))

    print(matrix)

    model.evaluate(x_test, y_test)

    model.save('model_example')
