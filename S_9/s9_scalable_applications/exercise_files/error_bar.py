import matplotlib.pyplot as plt
import numpy as np

time = np.array([18.271229934692382, 9.377594327926635, 7.810283184051514, 7.145597887039185, 8.072267198562622])
uncertainty = np.array([1.1988898984600331, 0.37526306876966986, 0.6605869485073566, 0.538575567881156, 1.176454830260331])
number_of_workers = np.array([1,2,3,4,5])

fig, ax = plt.subplots()
ax.errorbar(number_of_workers, time, xerr=0.02, yerr=uncertainty)
plt.ylabel('Time (s)')
plt.xlabel('Number of Workers')
plt.title('Time vs Number of Workers with Uncertainty')
plt.show()