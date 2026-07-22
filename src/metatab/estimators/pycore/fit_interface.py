"""
Module that contains the fit interface for the py classes.
The idea here is to provide different fit signatures using a mixin class.
See "__init__" file of pycore folder for more details about why this is first of all necessary.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from metatab.estimators.core.configurations import EarlyStopConfiguration

if TYPE_CHECKING:
    from metatab.metatab_utils.types import XType, YType



class FitInterfaceNoES:
    def fit(self, X: XType, y: YType):
        '''Fit on training data'''
        super().fit(X, y)
        return self


class FitInterfaceWithES:
    def fit(self, X, y, validation_set_size: float = 0.3):
        '''Fit on training data with early stop on validation set'''
        esc = EarlyStopConfiguration(validation_set_size=validation_set_size)
        super().fit(X, y, esc)
        return self


class FitInterfaceWithFullES:
    def fit(self, X, y, validation_set_size: float = 0.3, early_stop_rounds: int = 100):
        '''Fit on training data with early stop on validation set'''
        esc = EarlyStopConfiguration(early_stop_rounds, validation_set_size)
        super().fit(X, y, esc)
        return self
