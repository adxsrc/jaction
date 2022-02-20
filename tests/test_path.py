# Copyright (C) 2022 Chi-kwan Chan
# Copyright (C) 2022 Steward Observatory
#
# This file is part of jaction.
#
# Jaction is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jaction is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jaction.  If not, see <http://www.gnu.org/licenses/>.


from jaction import Path
from jax     import numpy as np


def test_path():

    def L(t, x, v):
        return np.sum(0.5 * v * v - 0.5 * x * x)

    x0 = np.array([[0.0,1.0], [1.0,0.0]])
    ns = Path(L, 0, x0, 1, atol=1e-5, rtol=0)

    x  = np.linspace(0, 10, num=101)
    y  = ns(x) # numerical solution

    assert np.max(abs(np.sin(x) - y[:,0,0])) < 1e-5
    assert np.max(abs(np.cos(x) - y[:,0,1])) < 1e-5
