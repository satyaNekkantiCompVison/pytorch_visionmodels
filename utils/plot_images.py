import matplotlib.pyplot as plt
import seaborn as sns


def plot_axis(train_loss=None, test_loss=None, train_accur = None, test_accur=None):
	fig, axs = plt.subplots(1,2,figsize=(18,8))
	axs[0].set_title('LOSS')
	axs[0].plot(train_loss, label='Train')
	axs[0].plot(test_loss, label='Test')
	axs[0].legend()
	axs[0].grid()

	axs[1].set_title('Accuracy')
	axs[1].plot(train_accur, label='Train')
	axs[1].plot(test_accur, label='Test')
	axs[1].legend()
	axs[1].grid()

	plt.show()