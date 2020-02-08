from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import io
import numpy as np
import sympy as sym

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


def f(X, q, b, c, n=2):
    Z = np.zeros(len(X))
    for i in range(len(X)):
        for j in range(int(n)):
            for k in range(int(n)):
                Z[i] += q[j][k] * X[i][j] * X[i][k]

        for j in range(int(n)):
            Z[i] += b[j] * X[i][j]

        Z[i] += c
    return Z


def f2(X, Y, q, b, c):
    Z = q[0][0]*X*X + q[0][1]*X*Y + q[1][0]*Y*X + q[1][1]*Y*Y + b[0]*X + b[1]*Y + int(c[0])
    return Z


def f_mesh(X, Y, q, b, c):
    Z = np.zeros(len(X))
    Z = q[0][0]*X*X + q[0][1]*X*Y + q[1][0]*Y*X + q[1][1]*Y*Y + b[0]*X + b[1]*Y + c
    return Z


def z_func(X_old, q, b, c, n=2):
    x, y, t = sym.symbols('x y t')

    X = sym.Matrix([[x, y]])

    T = sym.Matrix([[t]])

    df = sym.Matrix([[sym.diff(f2(x, y, q, b, c), x),
                      sym.diff(f2(x, y, q, b, c), y)]])
    z = X - t * df

    z = f2(z[0], z[1], q, b, c)
    z_diff = sym.diff(z, t)
    eqn = sym.Eq(z_diff)
    sol = sym.solve((eqn), (t))
    sym.expr = sol[0]
    sym.expr = sym.expr.subs([(x, X_old[0][0]), (y, X_old[0][1])])
    return sym.expr


def init():
    X1 = np.arange(-5, 5, 0.1)
    Y1 = np.arange(-5, 5, 0.1)
    Z1 = np.zeros(len(X1))

    X_new = np.zeros((100, 2))

    for i in range(len(X1)):
        X_new[i][0] = X1[i]
        X_new[i][1] = Y1[i]

    return X1, Y1, Z1, X_new


def create_figure(X1, Y1, Z1, x_list, y_list, q, b, c):
    X1, Y1 = np.meshgrid(X1, Y1)
    Z1 = f_mesh(X1, Y1, q, b, c)

    X, Y = zip(*x_list)
    Z = y_list

    ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    cs = plt.contour(X1, Y1, Z1)
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
    return plt


def grad_descent(X, X1, Y1, Y, q, b, c, eps=0.05, precision=0.0001, max_iter=200, n=2):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    dfr = np.zeros((1, 2))
    X_new[0][0] = 4.9
    X_new[0][1] = 4.9
    i = 0
    Xs = np.zeros((max_iter, 2))
    Ys = np.zeros(max_iter)
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)
    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        X_new = X_new - eps * dfr
        i += 1
        eps *= 0.99

    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], n, q, b, c)

        for j in range(max_iter - 1, i, -1):
            Xs = np.delete(Xs, j, axis=0)
            Ys = np.delete(Ys, j, axis=0)

    return Xs, Ys


def steepest(X, Y, q, b, c, precision=0.0001, max_iter=200, n=2):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    dfr = np.zeros((1, 2))
    X_new[0][0] = 4.9
    X_new[0][1] = 4.9
    i = 0
    Xs = np.zeros((max_iter, 2))
    Ys = np.zeros(max_iter)
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)

    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        eps = z_func(X_old, q, b, c)
        X_new = X_old - eps * dfr
        i += 1
    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        for j in range(max_iter - 1, i, -1):
            Xs = np.delete(Xs, j, axis=0)
            Ys = np.delete(Ys, j, axis=0)
    return Xs, Ys


def gd_with_momentum(X_new, X1, Y1, Z1, q, b, c, alpha=0.10, beta=0.9, precision=0.0001, max_iter=200, n=2):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    dfr = np.zeros((1, 2))
    X_new[0][0] = 4.9
    X_new[0][1] = 4.9
    i = 0
    Xs = np.zeros((max_iter, 2))
    Ys = np.zeros(max_iter)
    V = np.zeros((max_iter + 1, 2))
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)
    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        i += 1
        V[i] = beta * V[i - 1] + (1 - beta) * dfr
        X_new = X_new - alpha * V[i]
        alpha *= 0.99
    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        for j in range(max_iter - 1, i, -1):
            Xs = np.delete(Xs, j, axis=0)
            Ys = np.delete(Ys, j, axis=0)
    return Xs, Ys


def rmsprop (X_new, X1, Y1, Z1, q, b, c, alpha=0.10, beta=0.9, precision=0.0001, max_iter=200, n=2):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    dfr = np.zeros((1, 2))
    X_new[0][0] = 2.8
    X_new[0][1] = 4.9
    i = 0
    Xs = np.zeros((max_iter, 2))
    Ys = np.zeros(max_iter)
    S = np.zeros((max_iter + 1, 2))
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)
    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        X_old = X_new
        dfr[0][0] = df1.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        dfr[0][1] = df2.evalf(subs={x: X_old[0][0], y: X_old[0][1]})
        i += 1
        S[i] = beta * S[i - 1] + (1 - beta) * np.power(dfr, 2)
        X_new = X_new - alpha * dfr / np.sqrt(S[i])
        alpha *= 0.99
    print("Finished with {} step".format(i))
    if i < max_iter:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
        for j in range(max_iter - 1, i, -1):
            Xs = np.delete(Xs, j, axis=0)
            Ys = np.delete(Ys, j, axis=0)
    return Xs, Ys


def adam(X_new, X1, Y1, Z1, q, b, c, alpha=0.1, beta1=0.9, beta2=0.99, eps=0.000000001, precision=0.0001,
         max_iter=200, n=2):
    X_old = np.zeros((1, 2))
    X_new = np.zeros((1, 2))
    dfr = np.zeros((1, 2))
    X_new[0][0] = 2.8
    X_new[0][1] = 4.9
    i = 0
    Xs = np.zeros((max_iter, 2))
    Ys = np.zeros(max_iter)
    V = np.zeros((max_iter + 1, 2))
    S = np.zeros((max_iter + 1, 2))
    V_corr = np.zeros((1, 2))
    S_corr = np.zeros((1, 2))
    x, y = sym.symbols('x y')
    df1 = sym.diff(f2(x, y, q, b, c), x)
    df2 = sym.diff(f2(x, y, q, b, c), y)

    while np.all(abs(X_new - X_old)) > precision and max_iter > i:
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)
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
        Xs[i] = X_new
        Ys[i] = f2(X_new[0][0], X_new[0][1], q, b, c)

        for j in range(max_iter - 1, i, -1):
            Xs = np.delete(Xs, j, axis=0)
            Ys = np.delete(Ys, j, axis=0)
    return Xs, Ys


@app.route('/grad_des', methods=['POST'])
def gradient_descent():
    X1, Y1, Z1, X_new = init()

    eps = float(request.form['eps'])
    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = grad_descent(X_new, Z1, q, b, c, precision, max_iter)
    return plotter(X1, Y1, Z1, x_list, y_list, q, b, c)


@app.route('/steepest_des', methods=['POST'])
def steepest_descent():
    X1, Y1, Z1, X_new = init()
    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = steepest(X_new, Z1, q, b, c, precision, max_iter)
    return plotter(X1, Y1, Z1, x_list, y_list, q, b, c)


@app.route('/gdm', methods=['POST'])
def gd_with_m():
    X1, Y1, Z1, X_new = init()
    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])
    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = gd_with_momentum(X_new, X1, Y1, Z1, q, b, c, alpha, beta, precision, max_iter)
    return plotter(X1, Y1, Z1, x_list, y_list, q, b, c)


@app.route('/rmsprop', methods=['POST'])
def rms_prop():
    X1, Y1, Z1, X_new = init()
    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])
    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = rmsprop(X_new, X1, Y1, Z1, q, b, c, alpha, beta, precision, max_iter)
    return plotter(X1, Y1, Z1, x_list, y_list, q, b, c)


@app.route('/adam', methods=['POST'])
def ADAM():
    X1, Y1, Z1, X_new = init()
    precision = float(request.form['precision'])
    max_iter = int(request.form['max_iter'])
    alpha = float(request.form['alpha'])
    beta1 = float(request.form['beta1'])
    beta2 = float(request.form['beta2'])
    eps = float(request.form['eps'])
    q = [[request.form['q[0][0]'], request.form['q[0][1]']],
         [request.form['q[1][0]'], request.form['q[1][1]']]]
    b = [request.form['b[0]'], request.form['b[1]']]
    c = request.form['c']

    for i in range(2):
        q[i] = list(map(float, q[i]))
    b = list(map(float, b))
    c = list(map(float, c))

    Z1 = f(X_new, q, b, c)
    x_list, y_list = adam(X_new, X1, Y1, Z1, q, b, c, alpha, beta1, beta2, eps, precision, max_iter)
    return plotter(X1, Y1, Z1, x_list, y_list, q, b, c)


def plotter(X1, Y1, Z1, x_list, y_list, q, b, c):
    plt = create_figure(X1, Y1, Z1, x_list, y_list, q, b, c)
    output = io.BytesIO()
    plt.savefig(output, format='png')
    output.seek(0)
    return send_file(output, attachment_filename='plot.png', mimetype='image/png')

    # render_template('grad_des.html', user_image=output)


if __name__ == "__main__":
    app.run(debug=True)
