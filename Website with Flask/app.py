from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import imageio

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/grad_des')
def grad_des():
    return render_template('grad_des.html')


@app.route('/steepest_des')
def steepest_des():
    return render_template('steepest_des.html')


@app.route('/gdm')
def gdm():
    return render_template('gdm.html')


@app.route('/rmsprop')
def RMSprop():
    return render_template('rmsprop.html')


@app.route('/adam')
def adam_alg():
    return render_template('adam.html')


def f(x, q, b, c, n=2):
    z = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(int(n)):
            for k in range(int(n)):
                z[i] += q[j][k] * x[i][j] * x[i][k]

        for j in range(int(n)):
            z[i] += b[j] * x[i][j]

        z[i] += c
    return z


def f2(x, y, q, b, c):
    z = q[0][0] * x * x + q[0][1] * x * y + q[1][0] * y * x + q[1][1] * y * y + b[0] * x + b[1] * y + c[0]
    return z


def f_mesh(x, y, q, b, c):
    z = np.zeros(len(x))
    z = q[0][0] * x * x + q[0][1] * x * y + q[1][0] * y * x + q[1][1] * y * y + b[0] * x + b[1] * y + c[0]
    return z


def z_func(x_old, q, b, c):
    x, y, t = sym.symbols('x y t')

    x1 = sym.Matrix([[x, y]])

    t1 = sym.Matrix([[t]])

    df = sym.Matrix([[sym.diff(f2(x, y, q, b, c), x),
                      sym.diff(f2(x, y, q, b, c), y)]])
    z = x1 - t1 * df

    z = f2(z[0], z[1], q, b, c)
    z_diff = sym.diff(z, t)
    eqn = sym.Eq(z_diff)
    sol = sym.solve(eqn, t)
    sym.expr = sol[0]
    sym.expr = sym.expr.subs([(x, x_old[0][0]), (y, x_old[0][1])])
    return sym.expr


def init(start_x, end_x, inc_x, start_y, end_y, inc_y):
    X1 = np.arange(start_x, end_x, inc_x)
    Y1 = np.arange(start_y, end_y, inc_y)
    Z1 = np.zeros(len(X1))

    X_new = np.zeros((len(X1), 2))

    for i in range(len(X1)):
        X_new[i][0] = X1[i]
        X_new[i][1] = Y1[i]

    return X1, Y1, Z1, X_new


def make_gif(X1, Y1, Z1, x_list, y_list, q, b, c, gif):
    X1, Y1 = np.meshgrid(X1, Y1)
    Z1 = f_mesh(X1, Y1, q, b, c)

    x_list = np.delete(x_list, 0, axis=0)
    y_list = np.delete(y_list, 0, axis=0)

    frames = []

    for i in range(1, len(x_list)):
        X, Y = zip(*x_list[:i])
        Z = y_list[:i]

        ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        cs = plt.contour(X1, Y1, Z1)
        plt.suptitle('Iteration number: {}'.format(i), fontsize=14, fontweight='bold')
        plt.clabel(cs, inline=1, fontsize=10)
        colors = ['b', 'g', 'm', 'c', 'orange']
        for j in range(1, len(X)):
            ax[1].annotate('', xy=(X[j], Y[j]), xytext=(X[j - 1], Y[j - 1]),
                           arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                           va='center', ha='center')
        ax[1].scatter(X, Y, s=40, lw=0)
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[1].set_title('Minimizing function')
        plt.savefig('img.png')
        plt.close('all')

        new_frame = imageio.imread('img.png')
        frames.append(new_frame)

    imageio.mimsave('static/' + gif, frames)


def grad_descent(q, b, c, x0, y0, eps=0.05, precision=0.0001, max_iter=200):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    Y_new = np.zeros(1)
    dfr = np.zeros((1, 2))
    X_new[0][0] = x0
    X_new[0][1] = y0
    i = 0
    Xs = np.zeros((1, 2))
    Ys = np.zeros(1)
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)

    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        X_new = X_new - eps * dfr
        i += 1
        eps *= 0.99

    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
    return Xs, Ys


def steepest(q, b, c, x0, y0, precision=0.0001, max_iter=200):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    Y_new = np.zeros(1)
    dfr = np.zeros((1, 2))
    X_new[0][0] = x0
    X_new[0][1] = y0
    i = 0
    Xs = np.zeros((1, 2))
    Ys = np.zeros(1)
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)

    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        eps = z_func(X_old, q, b, c)
        X_new = X_old - eps * dfr
        i += 1
    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
    return Xs, Ys


def gd_with_momentum(q, b, c, x0, y0, alpha=0.10, beta=0.9, precision=0.0001, max_iter=200):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    Y_new = np.zeros(1)
    dfr = np.zeros((1, 2))
    X_new[0][0] = x0
    X_new[0][1] = y0
    i = 0
    Xs = np.zeros((1, 2))
    Ys = np.zeros(1)
    V = np.zeros((max_iter + 1, 2))
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)
    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        i += 1
        V[i] = beta * V[i - 1] + (1 - beta) * dfr
        X_new = X_new - alpha * V[i]
        alpha *= 0.99
    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
    return Xs, Ys


def rmsprop (q, b, c, x0, y0, alpha=0.10, beta=0.9, precision=0.0001, max_iter=200):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    Y_new = np.zeros(1)
    dfr = np.zeros((1, 2))
    X_new[0][0] = x0
    X_new[0][1] = y0
    i = 0
    Xs = np.zeros((1, 2))
    Ys = np.zeros(1)
    S = np.zeros((max_iter + 1, 2))
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)
    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        i += 1
        S[i] = beta * S[i - 1] + (1 - beta) * np.power(dfr, 2)
        X_new = X_new - alpha * dfr / np.sqrt(S[i])
        alpha *= 0.99
    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
    return Xs, Ys


def adam(q, b, c, x0, y0, alpha=0.1, beta1=0.9, beta2=0.99, eps=0.000000001, precision=0.0001, max_iter=200):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    Y_new = np.zeros(1)
    dfr = np.zeros((1, 2))
    X_new[0][0] = x0
    X_new[0][1] = y0
    i = 0
    Xs = np.zeros((1, 2))
    Ys = np.zeros(1)
    V = np.zeros((max_iter + 1, 2))
    S = np.zeros((max_iter + 1, 2))
    V_corr = np.zeros((1, 2))
    S_corr = np.zeros((1, 2))
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)

    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        i += 1
        V[i] = beta1 * V[i - 1] + (1 - beta1) * dfr  # momentum
        S[i] = beta2 * S[i - 1] + (1 - beta2) * np.power(dfr, 2)  # rmsprop
        V_corr = V[i] / (1 - np.power(beta1, i))
        S_corr = S[i] / (1 - np.power(beta2, i))
        X_new = X_new - alpha * V_corr / (np.sqrt(S_corr) + eps)
        alpha *= 0.99

    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs = np.append(Xs, X_new, axis=0)
        Y_new[0] = f2(X_new[0][0], X_new[0][1], q, b, c)
        Ys = np.append(Ys, Y_new, axis=0)
    return Xs, Ys


@app.route('/grad_des', methods=['POST'])
def gradient_descent():
    path = "grad_des.html"
    gif = "gradient_descent.gif"

    eps = float(request.form['eps'])
    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])

    startx = float(request.form['startx'])
    endx = float(request.form['endx'])
    incx = float(request.form['incx'])
    starty = float(request.form['starty'])
    endy = float(request.form['endy'])
    incy = float(request.form['incy'])
    x0 = float(request.form['x0'])
    y0 = float(request.form['y0'])

    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    X1, Y1, Z1, X_new = init(startx, endx, incx, starty, endy, incy)

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = grad_descent(q, b, c, x0, y0, eps, precision, max_iter)
    make_gif(X1, Y1, Z1, x_list, y_list, q, b, c, gif)
    return render_template(path)


@app.route('/steepest_des', methods=['POST'])
def steepest_descent():
    path = "steepest_des.html"
    gif = "steepest_descent.gif"

    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])

    startx = float(request.form['startx'])
    endx = float(request.form['endx'])
    incx = float(request.form['incx'])
    starty = float(request.form['starty'])
    endy = float(request.form['endy'])
    incy = float(request.form['incy'])
    x0 = float(request.form['x0'])
    y0 = float(request.form['y0'])

    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    X1, Y1, Z1, X_new = init(startx, endx, incx, starty, endy, incy)

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = steepest(q, b, c, x0, y0, precision, max_iter)
    make_gif(X1, Y1, Z1, x_list, y_list, q, b, c, gif)
    return render_template(path)


@app.route('/gdm', methods=['POST'])
def gd_with_m():
    path = "gdm.html"
    gif = "gdm.gif"

    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])

    startx = float(request.form['startx'])
    endx = float(request.form['endx'])
    incx = float(request.form['incx'])
    starty = float(request.form['starty'])
    endy = float(request.form['endy'])
    incy = float(request.form['incy'])
    x0 = float(request.form['x0'])
    y0 = float(request.form['y0'])

    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    X1, Y1, Z1, X_new = init(startx, endx, incx, starty, endy, incy)

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = gd_with_momentum(q, b, c, x0, y0, alpha, beta, precision, max_iter)
    make_gif(X1, Y1, Z1, x_list, y_list, q, b, c, gif)
    return render_template(path)


@app.route('/rmsprop', methods=['POST'])
def rms_prop():
    path = "rmsprop.html"
    gif = "rmsprop.gif"

    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])

    startx = float(request.form['startx'])
    endx = float(request.form['endx'])
    incx = float(request.form['incx'])
    starty = float(request.form['starty'])
    endy = float(request.form['endy'])
    incy = float(request.form['incy'])
    x0 = float(request.form['x0'])
    y0 = float(request.form['y0'])

    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    X1, Y1, Z1, X_new = init(startx, endx, incx, starty, endy, incy)

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = rmsprop(q, b, c, x0, y0, alpha, beta, precision, max_iter)
    make_gif(X1, Y1, Z1, x_list, y_list, q, b, c, gif)
    return render_template(path)


@app.route('/adam', methods=['GET', 'POST'])
def ADAM():
    path = "adam.html"
    gif = "adam.gif"

    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    alpha = float(request.form['alpha'])
    beta1 = float(request.form['beta1'])
    beta2 = float(request.form['beta2'])
    eps = float(request.form['eps'])

    startx = float(request.form['startx'])
    endx = float(request.form['endx'])
    incx = float(request.form['incx'])
    starty = float(request.form['starty'])
    endy = float(request.form['endy'])
    incy = float(request.form['incy'])
    x0 = float(request.form['x0'])
    y0 = float(request.form['y0'])

    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    X1, Y1, Z1, X_new = init(startx, endx, incx, starty, endy, incy)

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = adam(q, b, c, x0, y0, alpha, beta1, beta2, eps, precision, max_iter)
    make_gif(X1, Y1, Z1, x_list, y_list, q, b, c, gif)
    return render_template(path)


if __name__ == "__main__":
    app.run(debug=True)
