import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# line-plot
x = np.arange(1, 11)

print(x)

y1 = 2 * x
y2 = 3 * x

print(y1, y2)

plt.plot(x, y1, color='blue', linewidth=2, linestyle=':')
plt.plot(x, y2, color='orange', linewidth=2, linestyle=':')
plt.title("Line plot")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True)
plt.show()

# bar-plot
student = {"Sam": 30, "Bob": 50, "Julia": 70}

names = list(student.keys())
marks = list(student.values())

print(names, marks)

plt.bar(names, marks, color='yellow')
plt.title("Marks of Students")
plt.xlabel("Names of Students")
plt.ylabel("Marks")
plt.grid(True)
plt.show()

# scatter-plot
x = [4, 7, 3, 9, 1, 6]
y1 = [9, 1, 2, 6, 4, 9]
y2 = [4, 7, 8, 1, 2, 6]

plt.scatter(x, y1)
plt.scatter(x, y2, color='r')
plt.grid(True)
plt.show()

# histogram
ratings = pd.read_csv('../datasets/data/ml-latest-small/ratings.csv')

print(ratings.head())
plt.hist(ratings['rating'])
plt.show()

# pie-chart
fruits = ["apple", "mango", "orange", "banana"]
cost = [76, 45, 90, 85]

plt.pie(cost, labels=fruits, autopct='%0.1f%%', shadow=True)
plt.show()