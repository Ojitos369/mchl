# Python
import os
import json
import numpy as np
import pandas as pd
from apis import RedNeuronal

# Ojitos369
from ojitos369.utils import get_d

# User
from app.core.bases.apis import PostApi, GetApi

class HelloWorld(GetApi):
    def main(self):
        self.response = {
            'message': 'Hello World'
        }


class Test(GetApi):
    def main(self):
        self.response = {
            'message': 'Test'
        }

