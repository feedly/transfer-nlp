from transfer_nlp.plugins.config import register_plugin, ExperimentConfig, UnconfiguredItemsException
import unittest


@register_plugin
class Demo2:

    def __init__(self, simple_str:str):
        self.val = simple_str


@register_plugin
class Demo3:

    def __init__(self, simple_int:str):
        self.val = simple_int


@register_plugin
class Demo:

    def __init__(self, demo2, demo3: Demo3):
        self.demo2 = demo2
        self.demo3 = demo3

class RegistryTest(unittest.TestCase):

    def test_child_injection(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
            },
            'demo2': {
                '_name': 'Demo2'
            },
            'demo3': {
                '_name': 'Demo3'
            },
            'simple_str': 'dummy',
            'simple_int': 5
        }
        e = ExperimentConfig.from_json(experiment)

        self.assertTrue(isinstance(e['demo'], Demo))
        self.assertTrue(isinstance(e['demo2'], Demo2))
        self.assertTrue(isinstance(e['demo3'], Demo3))

        self.assertEqual(e['demo2'].val, 'dummy')
        self.assertEqual(e['demo3'].val, e['demo3'].val, 5)

    def test_child_named_injection(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
                'demo3': 'demo3a'

            },
            'demo2': {
                '_name': 'Demo2'
            },
            'demo3': {
                '_name': 'Demo3'
            },
            'demo3a': {
                '_name': 'Demo3',
                'simple_int': 'simple_inta'
            },
            'simple_str': 'dummy',
            'simple_int': 5,
            'simple_inta': 6,
        }
        e = ExperimentConfig.from_json(experiment)

        self.assertTrue(isinstance(e['demo'], Demo))
        self.assertTrue(isinstance(e['demo2'], Demo2))
        self.assertTrue(isinstance(e['demo3'], Demo3))
        self.assertTrue(isinstance(e['demo3a'], Demo3))

        self.assertEqual(e['demo2'].val, 'dummy')
        self.assertEqual(e['demo3'].val, 5)
        self.assertEqual(e['demo3a'].val,  6)
        self.assertEqual(e['demo'].demo3.val, 6)

    def test_literal_injection(self):
        experiment = {
            'demo2': {
                '_name': 'Demo2',
                'simple_str_': 'dummy'
            },
            'demo3': {
                '_name': 'Demo3',
                'simple_int_': 5
            }
        }
        e = ExperimentConfig.from_json(experiment)

        self.assertTrue(isinstance(e['demo2'], Demo2))
        self.assertTrue(isinstance(e['demo3'], Demo3))

        self.assertEqual(e['demo2'].val, 'dummy')
        self.assertEqual(e['demo3'].val, 5)

    def test_unconfigured(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
            },
            'demo2': {
                '_name': 'Demo2',
            },
            'demo3': {
                '_name': 'Demo3',
            }
        }

        try:
            ExperimentConfig.from_json(experiment)
            self.fail()
        except UnconfiguredItemsException as e:
            self.assertEqual(3, len(e.items))
            self.assertEqual({'demo2', 'demo3'}, e.items['demo'])
            self.assertEqual({'simple_str'}, e.items['demo2'])
            self.assertEqual({'simple_int'}, e.items['demo3'])
