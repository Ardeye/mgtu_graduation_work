from flask import Flask, request, render_template
from keras.models import load_model
import pandas as pd
model = load_model('neural_net')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def processing():
    show_result = ''
    if request.method == 'POST':
        data = request.form
        try:
            data_to_float = {k: [float(v)] for k, v in data.items()}
            process = pd.DataFrame(data=data_to_float)
            result = model.predict(process)[0][0]
            show_result = f'Соотношение матрица-наполнитель = {result}'
        except ValueError:
            print('input error')

    return render_template('app.html', result=show_result)


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
