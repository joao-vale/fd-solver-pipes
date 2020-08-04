import numpy as np
import matplotlib.pyplot as plt
from plot_style import style_cmyk
from matplotlib import cm

style_cmyk()


def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * ((1 / dt) * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx) +
                                       (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy)) -
                           ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx)) ** 2 -
                           2 * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dy) *
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dx)) -
                           ((v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy)) ** 2)

    # Periodic BC Pressure @ x = 2
    b[-1, 1:-1] = rho * ((1 / dt) * ((u[0, 1:-1] - u[-2, 1:-1]) / (2 * dx) +
                                     (v[-1, 2:] - v[-1, 0:-2]) / (2 * dy)) -
                         ((u[0, 1:-1] - u[-2, 1:-1]) / (2 * dx)) ** 2 -
                         2 * ((u[-1, 2:] - u[-1, 0:-2]) / (2 * dy) *
                              (v[0, 1:-1] - v[-2, 1:-1]) / (2 * dx)) -
                         ((v[-1, 2:] - v[-1, 0:-2]) / (2 * dy)) ** 2)

    # Periodic BC Pressure @ x = 0
    b[0, 1:-1] = rho * ((1 / dt) * ((u[1, 1:-1] - u[-1, 1:-1]) / (2 * dx) +
                                    (v[0, 2:] - v[0, 0:-2]) / (2 * dy)) -
                        ((u[1, 1:-1] - u[-1, 1:-1]) / (2 * dx)) ** 2 -
                        2 * ((u[0, 2:] - u[0, 0:-2]) / (2 * dy) *
                             (v[1, 1:-1] - v[-1, 1:-1]) / (2 * dx)) -
                        ((v[0, 2:] - v[0, 0:-2]) / (2 * dy)) ** 2)
    return b


def pressure_poison_periodic(p, dx, dy):
    pn = np.empty_like(p)

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dy ** 2 +
                          (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) -
                         ((dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[-1, 1:-1] = (((pn[0, 1:-1] + pn[-2, 1:-1]) * dy ** 2 +
                        (pn[-1, 2:] + pn[-1, 0:-2]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) -
                       ((dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))) * b[-1, 1:-1])

        # Periodic BC Pressure @ x = 0
        p[0, 1:-1] = (((pn[1, 1:-1] + pn[-1, 1:-1]) * dy ** 2 +
                       (pn[0, 2:] + pn[0, 0:-2]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) -
                      ((dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))) * b[0, 1:-1])

        # Wall boundary conditions for pressure
        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]

    return p


np.set_printoptions(threshold=np.inf)

nx = 101
ny = 101
nt = 10
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1
nu = .1
F = 1
dt = .001

u = np.zeros((nx, ny))
un = np.zeros((nx, ny))

v = np.zeros((nx, ny))
vn = np.zeros((nx, ny))

p = np.ones((nx, ny))
pn = np.ones((nx, ny))

b = np.zeros((nx, ny))

udiff = 1
stepcount = 0

while udiff > .001:
    un = u.copy()
    vn = v.copy()
    b = build_up_b(rho, dt, dx, dy, u, v)
    p = pressure_poison_periodic(p, dx, dy)
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * (dt / dx) * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     vn[1:-1, 1:-1] * (dt / dy) * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     (dt / (rho * 2 * dx)) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * ((dt / (dx ** 2)) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +
                           (dt / (dy ** 2)) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])) +
                     dt * F)
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * (dt / dx) * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     vn[1:-1, 1:-1] * (dt / dy) * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     (dt / (rho * 2 * dy)) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * ((dt / (dx ** 2)) * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) +
                           (dt / (dy ** 2)) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])))

    # Periodic BC u @ x = 2
    u[-1, 1:-1] = (un[-1, 1:-1] -
                   un[-1, 1:-1] * (dt / dx) * (un[-1, 1:-1] - un[-2, 1:-1]) -
                   vn[-1, 1:-1] * (dt / dy) * (un[-1, 1:-1] - un[-1, 0:-2]) -
                   (dt / (rho * 2 * dx)) * (p[0, 1:-1] - p[-2, 1:-1]) +
                   nu * ((dt / (dx ** 2)) * (un[0, 1:-1] - 2 * un[-1, 1:-1] + un[-2, 1:-1]) +
                         (dt / (dy ** 2)) * (un[-1, 2:] - 2 * un[-1, 1:-1] + un[-1, 0:-2])) +
                   dt * F)
    # Periodic BC u @ x = 0
    u[0, 1:-1] = (un[0, 1:-1] -
                  un[0, 1:-1] * (dt / dx) * (un[0, 1:-1] - un[-1, 1:-1]) -
                  vn[0, 1:-1] * (dt / dy) * (un[0, 1:-1] - un[0, 0:-2]) -
                  (dt / (rho * 2 * dx)) * (p[1, 1:-1] - p[-1, 1:-1]) +
                  nu * ((dt / (dx ** 2)) * (un[1, 1:-1] - 2 * un[0, 1:-1] + un[-1, 1:-1]) +
                        (dt / (dy ** 2)) * (un[0, 2:] - 2 * un[0, 1:-1] + un[0, 0:-2])) +
                  dt * F)

    # Periodic BC v @ x = 2
    v[-1, 1:-1] = (vn[-1, 1:-1] -
                   un[-1, 1:-1] * (dt / dx) * (vn[-1, 1:-1] - vn[-2, 1:-1]) -
                   vn[-1, 1:-1] * (dt / dy) * (vn[-1, 1:-1] - vn[-1, 0:-2]) -
                   (dt / (rho * 2 * dy)) * (p[0, 1:-1] - p[-2, 1:-1]) +
                   nu * ((dt / (dx ** 2)) * (vn[0, 1:-1] - 2 * vn[-1, 1:-1] + vn[-2, 1:-1]) +
                         (dt / (dy ** 2)) * (vn[-1, 2:] - 2 * vn[-1, 1:-1] + vn[-1, 0:-2])))
    # Periodic BC v @ x = 2
    v[0, 1:-1] = (vn[0, 1:-1] -
                  un[0, 1:-1] * (dt / dx) * (vn[0, 1:-1] - vn[-1, 1:-1]) -
                  vn[0, 1:-1] * (dt / dy) * (vn[0, 1:-1] - vn[0, 0:-2]) -
                  (dt / (rho * 2 * dy)) * (p[1, 1:-1] - p[-1, 1:-1]) +
                  nu * ((dt / (dx ** 2)) * (vn[1, 1:-1] - 2 * vn[0, 1:-1] + vn[-1, 1:-1]) +
                        (dt / (dy ** 2)) * (vn[0, 2:] - 2 * vn[0, 1:-1] + vn[0, 0:-2])))

    # wall bc : u, v = 0 @ y = 0, 2
    u[:, 0] = 0
    u[:, -1] = 0
    v[:, 0] = 0
    v[:, -1] = 0

    udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
    # udiff = 0.000001
    stepcount += 1
    # print(u)
fig = plt.figure()
V = np.sqrt(u ** 2 + v ** 2)
# plt.quiver(Y[::3, ::3], X[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.contourf(Y, X, V, 255, cmap=cm.jet)
plt.show()
