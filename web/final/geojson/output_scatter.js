var scatter_data_orginal = {'hours': ['0.0%', '3.0%', '6.1%', '9.1%', '12.1%', '15.2%', '18.2%', '21.2%', '24.2%', '27.3%', '30.3%', '33.3%', '36.4%', '39.4%', '42.4%', '45.5%', '48.5%', '51.5%', '54.5%', '57.6%', '60.6%', '63.6%', '66.7%', '69.7%', '72.7%', '75.8%', '78.8%', '81.8%', '84.8%', '87.9%', '90.9%', '93.9%', '97.0%'], 'days': ['building;edifice', 'bed', 'grass', 'table', 'column;pillar', 'chest;of;drawers;chest;bureau;dresser', 'counter', 'grandstand;covered;stand', 'countertop', 'towel', 'minibike;motorbike', 'fan', 'clock'], 'scatter_data': [[2, 0, 10.0], [3, 0, 10.0], [5, 0, 10.0], [8, 0, 10.0], [1, 0, 10.0], [9, 0, 10.0], [6, 0, 10.0], [11, 0, 5.83], [7, 0, 5.7], [4, 0, 1.53], [12, 0, 1.53], [10, 0, 1.53], [0, 0, 1.53], [0, 1, 10.0], [11, 1, 10.0], [10, 1, 10.0], [6, 1, 10.0], [5, 1, 10.0], [2, 1, 10.0], [7, 1, 10.0], [3, 1, 10.0], [8, 1, 9.94], [1, 1, 7.46], [4, 1, 2.84], [12, 1, 1.53], [9, 1, 1.53], [0, 2, 10.0], [2, 2, 10.0], [8, 2, 10.0], [10, 2, 10.0], [5, 2, 10.0], [1, 2, 10.0], [6, 2, 9.99], [3, 2, 9.53], [7, 2, 7.18], [12, 2, 4.92], [9, 2, 3.37], [4, 2, 1.53], [11, 2, 1.53], [2, 3, 10.0], [1, 3, 10.0], [8, 3, 10.0], [5, 3, 10.0], [11, 3, 10.0], [7, 3, 9.89], [4, 3, 3.48], [6, 3, 3.22], [3, 3, 2.78], [9, 3, 2.4], [12, 3, 1.53], [10, 3, 1.53], [0, 3, 1.53], [1, 4, 10.0], [6, 4, 10.0], [3, 4, 10.0], [2, 4, 10.0], [7, 4, 10.0], [5, 4, 10.0], [8, 4, 10.0], [11, 4, 10.0], [10, 4, 2.17], [4, 4, 1.53], [12, 4, 1.53], [9, 4, 1.53], [0, 4, 1.53], [8, 5, 10.0], [2, 5, 10.0], [5, 5, 10.0], [11, 5, 10.0], [3, 5, 10.0], [6, 5, 10.0], [1, 5, 9.96], [7, 5, 8.66], [4, 5, 1.53], [12, 5, 1.53], [10, 5, 1.53], [9, 5, 1.53], [0, 5, 1.53], [1, 6, 10.0], [5, 6, 10.0], [8, 6, 10.0], [2, 6, 10.0], [9, 6, 10.0], [7, 6, 9.99], [3, 6, 9.97], [6, 6, 9.59], [11, 6, 7.61], [10, 6, 2.38], [4, 6, 1.53], [12, 6, 1.53], [0, 6, 1.53], [8, 7, 10.0], [5, 7, 10.0], [2, 7, 9.99], [1, 7, 9.97], [6, 7, 7.37], [7, 7, 6.06], [3, 7, 3.67], [4, 7, 1.53], [12, 7, 1.53], [10, 7, 1.53], [9, 7, 1.53], [11, 7, 1.53], [0, 7, 1.53], [9, 8, 10.0], [5, 8, 10.0], [8, 8, 10.0], [3, 8, 10.0], [2, 8, 10.0], [11, 8, 9.97], [7, 8, 9.24], [6, 8, 8.59], [10, 8, 6.69], [1, 8, 5.23], [4, 8, 1.53], [12, 8, 1.53], [0, 8, 1.53], [8, 9, 10.0], [2, 9, 10.0], [5, 9, 10.0], [3, 9, 9.98], [6, 9, 9.92], [7, 9, 9.83], [11, 9, 9.78], [4, 9, 1.53], [1, 9, 1.53], [12, 9, 1.53], [10, 9, 1.53], [9, 9, 1.53], [0, 9, 1.53], [7, 10, 10.0], [8, 10, 10.0], [6, 10, 10.0], [5, 10, 10.0], [11, 10, 10.0], [2, 10, 10.0], [1, 10, 8.26], [3, 10, 6.68], [4, 10, 2.23], [12, 10, 1.53], [10, 10, 1.53], [9, 10, 1.53], [0, 10, 1.53], [1, 11, 10.0], [8, 11, 10.0], [5, 11, 10.0], [2, 11, 10.0], [11, 11, 10.0], [3, 11, 10.0], [7, 11, 5.79], [4, 11, 1.53], [6, 11, 1.53], [12, 11, 1.53], [10, 11, 1.53], [9, 11, 1.53], [0, 11, 1.53], [2, 12, 10.0], [8, 12, 10.0], [1, 12, 10.0], [3, 12, 10.0], [5, 12, 10.0], [11, 12, 6.51], [6, 12, 3.52], [7, 12, 2.31], [4, 12, 1.53], [12, 12, 1.53], [10, 12, 1.53], [9, 12, 1.53], [0, 12, 1.53], [2, 13, 10.0], [11, 13, 10.0], [5, 13, 10.0], [8, 13, 10.0], [1, 13, 10.0], [6, 13, 9.99], [3, 13, 3.94], [7, 13, 2.44], [10, 13, 2.38], [4, 13, 1.53], [12, 13, 1.53], [9, 13, 1.53], [0, 13, 1.53], [8, 14, 10.0], [5, 14, 10.0], [6, 14, 9.99], [11, 14, 9.26], [2, 14, 8.97], [7, 14, 7.78], [4, 14, 3.21], [10, 14, 2.18], [1, 14, 1.53], [12, 14, 1.53], [3, 14, 1.53], [9, 14, 1.53], [0, 14, 1.53], [11, 15, 10.0], [2, 15, 10.0], [3, 15, 10.0], [5, 15, 10.0], [8, 15, 10.0], [1, 15, 9.96], [6, 15, 9.87], [10, 15, 6.82], [7, 15, 3.2], [4, 15, 1.53], [12, 15, 1.53], [9, 15, 1.53], [0, 15, 1.53], [6, 16, 10.0], [5, 16, 10.0], [8, 16, 10.0], [11, 16, 10.0], [7, 16, 10.0], [2, 16, 10.0], [1, 16, 10.0], [3, 16, 8.97], [10, 16, 4.24], [4, 16, 1.53], [12, 16, 1.53], [9, 16, 1.53], [0, 16, 1.53], [4, 17, 10.0], [3, 17, 10.0], [6, 17, 10.0], [7, 17, 10.0], [1, 17, 10.0], [5, 17, 10.0], [8, 17, 10.0], [2, 17, 9.99], [11, 17, 9.97], [10, 17, 4.18], [12, 17, 1.53], [9, 17, 1.53], [0, 17, 1.53], [11, 18, 10.0], [10, 18, 10.0], [3, 18, 10.0], [5, 18, 10.0], [8, 18, 10.0], [2, 18, 10.0], [6, 18, 9.87], [1, 18, 9.35], [4, 18, 6.62], [7, 18, 5.94], [12, 18, 1.53], [9, 18, 1.53], [0, 18, 1.53], [5, 19, 10.0], [8, 19, 10.0], [2, 19, 10.0], [1, 19, 10.0], [3, 19, 10.0], [11, 19, 9.91], [6, 19, 9.85], [7, 19, 9.31], [4, 19, 3.43], [10, 19, 2.13], [12, 19, 1.53], [9, 19, 1.53], [0, 19, 1.53], [8, 20, 10.0], [5, 20, 10.0], [2, 20, 10.0], [11, 20, 10.0], [1, 20, 9.98], [3, 20, 9.55], [6, 20, 8.92], [7, 20, 2.35], [4, 20, 1.53], [12, 20, 1.53], [10, 20, 1.53], [9, 20, 1.53], [0, 20, 1.53], [8, 21, 10.0], [9, 21, 10.0], [1, 21, 10.0], [5, 21, 10.0], [2, 21, 10.0], [11, 21, 9.98], [3, 21, 8.81], [6, 21, 6.47], [7, 21, 4.42], [12, 21, 2.44], [4, 21, 1.53], [10, 21, 1.53], [0, 21, 1.53], [11, 22, 10.0], [2, 22, 10.0], [5, 22, 10.0], [8, 22, 10.0], [6, 22, 9.38], [7, 22, 3.12], [3, 22, 2.19], [4, 22, 1.53], [1, 22, 1.53], [12, 22, 1.53], [10, 22, 1.53], [9, 22, 1.53], [0, 22, 1.53], [8, 23, 10.0], [11, 23, 10.0], [2, 23, 10.0], [5, 23, 10.0], [3, 23, 9.29], [1, 23, 6.11], [7, 23, 5.67], [6, 23, 1.76], [4, 23, 1.53], [12, 23, 1.53], [10, 23, 1.53], [9, 23, 1.53], [0, 23, 1.53], [2, 24, 10.0], [1, 24, 10.0], [3, 24, 10.0], [5, 24, 10.0], [8, 24, 10.0], [6, 24, 10.0], [11, 24, 9.92], [7, 24, 9.62], [10, 24, 4.58], [0, 24, 2.6], [4, 24, 1.53], [12, 24, 1.53], [9, 24, 1.53], [8, 25, 10.0], [2, 25, 10.0], [5, 25, 10.0], [6, 25, 9.95], [11, 25, 4.68], [10, 25, 2.43], [4, 25, 1.53], [1, 25, 1.53], [12, 25, 1.53], [3, 25, 1.53], [9, 25, 1.53], [7, 25, 1.53], [0, 25, 1.53], [2, 26, 10.0], [6, 26, 10.0], [10, 26, 10.0], [5, 26, 10.0], [8, 26, 10.0], [4, 26, 9.99], [1, 26, 9.82], [11, 26, 9.26], [3, 26, 5.06], [9, 26, 2.71], [12, 26, 2.24], [7, 26, 1.9], [0, 26, 1.53], [3, 27, 10.0], [1, 27, 10.0], [6, 27, 10.0], [7, 27, 10.0], [5, 27, 10.0], [2, 27, 10.0], [10, 27, 10.0], [8, 27, 10.0], [11, 27, 9.45], [12, 27, 6.3], [4, 27, 1.53], [9, 27, 1.53], [0, 27, 1.53], [8, 28, 10.0], [1, 28, 10.0], [5, 28, 10.0], [3, 28, 10.0], [10, 28, 10.0], [7, 28, 9.99], [2, 28, 9.91], [11, 28, 9.78], [6, 28, 9.75], [12, 28, 8.09], [4, 28, 1.53], [9, 28, 1.53], [0, 28, 1.53], [2, 29, 10.0], [8, 29, 10.0], [5, 29, 10.0], [4, 29, 10.0], [11, 29, 10.0], [0, 29, 9.99], [6, 29, 9.9], [1, 29, 9.52], [3, 29, 7.45], [10, 29, 6.97], [9, 29, 6.39], [7, 29, 6.31], [12, 29, 1.53], [2, 30, 10.0], [8, 30, 10.0], [5, 30, 10.0], [6, 30, 10.0], [10, 30, 9.93], [4, 30, 8.85], [7, 30, 6.78], [11, 30, 6.6], [1, 30, 3.87], [9, 30, 3.14], [3, 30, 2.62], [12, 30, 1.53], [0, 30, 1.53], [6, 31, 10.0], [7, 31, 10.0], [9, 31, 10.0], [3, 31, 10.0], [2, 31, 10.0], [8, 31, 10.0], [11, 31, 10.0], [5, 31, 10.0], [1, 31, 4.37], [4, 31, 1.53], [12, 31, 1.53], [10, 31, 1.53], [0, 31, 1.53], [8, 32, 10.0], [11, 32, 10.0], [5, 32, 10.0], [6, 32, 10.0], [2, 32, 10.0], [9, 32, 9.98], [10, 32, 7.92], [1, 32, 7.06], [7, 32, 3.79], [4, 32, 1.53], [12, 32, 1.53], [3, 32, 1.53], [0, 32, 1.53]]} 
		