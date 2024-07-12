import matplotlib
import matplotlib.pyplot as plt

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        # print(shell)
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def plt_show():
    if is_notebook():
        plt.show()
