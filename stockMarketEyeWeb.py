from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired, NumberRange
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

class InputForm(FlaskForm):
    symbol = StringField('Aktiensymbol', validators=[DataRequired()])
    start_date = StringField('Startdatum (YYYY-MM-DD)', default='2017-07-01', validators=[DataRequired()])
    end_date = StringField('Enddatum (YYYY-MM-DD)', default=datetime.now().strftime('%Y-%m-%d'), validators=[DataRequired()])
    future_days = IntegerField('Zukunftstage', default=90, validators=[DataRequired(), NumberRange(min=1, max=365)])
    submit = SubmitField('Vorhersage')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    plot_url = None
    if form.validate_on_submit():
        stock_data = yf.download(form.symbol.data, start=form.start_date.data, end=form.end_date.data)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))

        def create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 100
        X, y = create_dataset(scaled_data, time_step)
        
        train_size = 0.8
        X_train, X_test = X[:int(X.shape[0]*train_size)], X[int(X.shape[0]*train_size):]
        y_train, y_test = y[:int(y.shape[0]*train_size)], y[int(y.shape[0]*train_size):]
        
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(units=64),
            Dense(units=64),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Neue Vorhersagen für die angeforderten Zukunftstage
        new_predictions = model.predict(X_test[-form.future_days.data:])
        new_predictions = scaler.inverse_transform(new_predictions)

        original_data = stock_data['Close'].values
        predicted_data = np.empty_like(original_data)
        predicted_data[:] = np.nan
        predicted_data[-len(predictions):] = predictions.reshape(-1)
        predicted_data = np.append(predicted_data, new_predictions)

        plt.figure(figsize=(10, 5))
        plt.plot(original_data, label='Original Data')
        plt.plot(np.arange(len(original_data) - len(predictions), len(original_data)), predictions, label='Predicted Data')
        plt.plot(np.arange(len(original_data), len(original_data) + len(new_predictions)), new_predictions, label='Future Predictions')
        plt.legend()
        plt.title('Stock Price Prediction including Future Predictions')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', form=form, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
