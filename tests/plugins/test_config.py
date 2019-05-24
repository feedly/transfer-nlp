import unittest
from pathlib import Path
from typing import List, Any, Dict

from transfer_nlp.plugins.config import register_plugin, UnconfiguredItemsException, ExperimentConfig, BadParameter, UnknownPluginException


@register_plugin
class DemoWithVal:

    def __init__(self, val: Any):
        self.val = val


@register_plugin
class DemoWithStr:

    def __init__(self, strval: str):
        self.strval = strval


@register_plugin
def demo_method_with_str(str_val: str):
    return DemoWithStr(strval=str_val)


@register_plugin
class DemoWithInt:

    def __init__(self, intval: str):
        self.intval = intval


@register_plugin
class DemoDefaults:

    def __init__(self, strval: str, intval1: int = 5, intval2: int = None):
        self.strval = strval
        self.intval1 = intval1
        self.intval2 = intval2


@register_plugin
class DemoComplexDefaults:

    def __init__(self, strval: str, obj: DemoDefaults = None):  # use different param and property names as additonal check
        self.simple = strval
        self.complex = obj


@register_plugin
class Demo:

    def __init__(self, demo2, demo3):
        self.demo2 = demo2
        self.demo3 = demo3


@register_plugin
class DemoWithConfig:

    def __init__(self, demo2, intval: int, experiment_config):
        self.demo2 = demo2
        self.intval = intval
        self.experiment_config = experiment_config


@register_plugin
class DemoA:
    def __init__(self, simple_int: int, attra: int = None):
        self.simple_int = simple_int
        self.attra = attra


@register_plugin
class DemoB:

    def __init__(self, demoa: DemoA, attrb: int = 2):
        self.demoa = demoa
        self.attrb = attrb


@register_plugin
class DemoC:
    def __init__(self, demob: DemoB, attrc: int = 3):
        self.demob = demob
        self.attrc = attrc


@register_plugin
class DemoWithList:
    def __init__(self, children: List[Any], simple_int: int = 3):
        self.children = children
        self.simple_int = simple_int


@register_plugin
class DemoWithDict:
    def __init__(self, children: Dict[str, Any], simple_int: int = 3):
        self.children = children
        self.simple_int = simple_int


@register_plugin
class Pipeline:

    def __init__(self, steps: List):
        self.steps = steps


class RegistryTest(unittest.TestCase):

    def test_recursive_definition(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
                'demo2': {
                    '_name': 'DemoWithStr',
                    'strval': 'foo'
                },
                'demo3': {
                    '_name': 'DemoWithInt',
                    'intval': 2
                }
            }
        }
        e = ExperimentConfig(experiment)
        self.assertIsInstance(e.experiment['demo'].demo2, DemoWithStr)
        self.assertIsInstance(e.experiment['demo'].demo3, DemoWithInt)
        self.assertEqual(e.experiment['demo'].demo2.strval, 'foo')
        self.assertEqual(e.experiment['demo'].demo3.intval, 2)
        self.assertEqual(e.factories.keys(), {'demo.demo2', 'demo.demo3', 'demo'})

    def test_child_injection(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
            },
            'demo2': {
                '_name': 'DemoWithStr'
            },
            'demo3': {
                '_name': 'DemoWithInt'
            },
            'strval': 'dummy',
            'intval': 5
        }
        e = ExperimentConfig(experiment)

        self.assertTrue(isinstance(e['demo'], Demo))
        self.assertTrue(isinstance(e['demo2'], DemoWithStr))
        self.assertTrue(isinstance(e['demo3'], DemoWithInt))

        self.assertEqual(e['demo2'].strval, 'dummy')
        self.assertEqual(e['demo3'].intval, 5)

    def test_child_named_injection(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
                'demo3': '$demo3a'

            },
            'demo2': {
                '_name': 'DemoWithStr'
            },
            'demo3': {
                '_name': 'DemoWithInt'
            },
            'demo3a': {
                '_name': 'DemoWithInt',
                'intval': '$simple_inta'
            },
            'strval': 'dummy',
            'intval': 5,
            'simple_inta': 6,
        }
        e = ExperimentConfig(experiment)

        self.assertTrue(isinstance(e['demo'], Demo))
        self.assertTrue(isinstance(e['demo2'], DemoWithStr))
        self.assertTrue(isinstance(e['demo3'], DemoWithInt))
        self.assertTrue(isinstance(e['demo3a'], DemoWithInt))

        self.assertEqual(e['demo2'].strval, 'dummy')
        self.assertEqual(e['demo3'].intval, 5)
        self.assertEqual(e['demo3a'].intval, 6)
        self.assertEqual(e['demo'].demo3.intval, 6)

    def test_env(self):
        experiment = {
            'path': "$HOME/foo/bar",
            'path2': "$HOMEPATH/foo/bar",
            'data': {
                '_name': "DemoWithStr",
                'strval': "$HOME/foo/bar/bis"
            },
            'data2': {
                '_name': "DemoDefaults",
                'strval': "foo",
                'intval1': "$SVAL"
            }
        }
        e = ExperimentConfig(experiment, HOME='/tmp', HOMEPATH=Path('/tmp2'), SVAL=7)
        self.assertEqual(e['path'], '/tmp/foo/bar')
        self.assertEqual(e['path2'], '/tmp2/foo/bar')

        self.assertEqual(e['data'].strval, '/tmp/foo/bar/bis')

        self.assertEqual(e['data2'].strval, 'foo')
        self.assertEqual(e['data2'].intval1, 7)
        self.assertIsNone(e['data2'].intval2)

    def test_literal_injection(self):
        experiment = {
            'demo2': {
                '_name': 'DemoWithStr',
                'strval': 'dummy'
            },
            'demo3': {
                '_name': 'DemoWithInt',
                'intval': 5
            }
        }
        e = ExperimentConfig(experiment)

        self.assertTrue(isinstance(e['demo2'], DemoWithStr))
        self.assertTrue(isinstance(e['demo3'], DemoWithInt))

        self.assertEqual(e['demo2'].strval, 'dummy')
        self.assertEqual(e['demo3'].intval, 5)

    def test_unconfigured(self):
        experiment = {
            'demo': {
                '_name': 'Demo',
            },
            'demo2': {
                '_name': 'DemoWithStr',
            },
            'demo3': {
                '_name': 'DemoWithInt',
            }
        }

        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnconfiguredItemsException as e:
            self.assertEqual(3, len(e.items))
            self.assertEqual({'demo2', 'demo3'}, e.items['demo'])
            self.assertEqual({'strval'}, e.items['demo2'])
            self.assertEqual({'intval'}, e.items['demo3'])

    def test_defaults(self):
        experiment = {
            'demoa': {
                '_name': 'DemoDefaults',
                'strval': 'a',
            },
            'demob': {
                '_name': 'DemoDefaults',
                'strval': 'b',
                'intval1': 1
            },
            'democ': {
                '_name': 'DemoDefaults',
                'strval': 'c',
                'intval2': 2
            },
            'demod': {
                '_name': 'DemoDefaults',
                'strval': 'd',
                'intval1': 3,
                'intval2': 4
            },
            'demoe': {
                '_name': 'DemoDefaults',
                'strval': 'e',
                'intval1': None,
                'intval2': None
            },
        }
        e = ExperimentConfig(experiment)

        for c in 'abcde':
            self.assertTrue(isinstance(e[f'demo{c}'], DemoDefaults))
            self.assertEqual(e[f'demo{c}'].strval, c)

        self.assertEqual(e['demoa'].intval1, 5)
        self.assertEqual(e['demoa'].intval2, None)

        self.assertEqual(e['demob'].intval1, 1)
        self.assertEqual(e['demob'].intval2, None)

        self.assertEqual(e['democ'].intval1, 5)
        self.assertEqual(e['democ'].intval2, 2)

        self.assertEqual(e['demod'].intval1, 3)
        self.assertEqual(e['demod'].intval2, 4)

        self.assertEqual(e['demoe'].intval1, None)
        self.assertEqual(e['demoe'].intval2, None)

    def test_complex_defaults(self):
        ### test that demod gets created first and then is used to create demo instead of the None default
        experiment = {
            'demo': {
                '_name': 'DemoComplexDefaults',
                'strval': 'foo'
            },
            'obj': {
                '_name': 'DemoDefaults',
                'strval': 'bar',
                'intval1': 20
            }
        }
        e = ExperimentConfig(experiment)

        self.assertTrue(isinstance(e['demo'], DemoComplexDefaults))
        self.assertTrue(isinstance(e['obj'], DemoDefaults))

        self.assertEqual(e['obj'].strval, 'bar')
        self.assertEqual(e['obj'].intval1, 20)
        self.assertEqual(e['obj'].intval2, None)

        self.assertEqual(e['demo'].simple, 'foo')
        self.assertEqual(e['demo'].complex.strval, 'bar')
        self.assertEqual(e['demo'].complex.intval1, 20)
        self.assertEqual(e['demo'].complex.intval2, None)

    def test_with_config(self):
        experiment = {
            'demo2': {
                '_name': 'DemoWithInt'
            },
            'with_config': {
                '_name': 'DemoWithConfig',
                'intval': 10
            },
            'intval': 5
        }

        e = ExperimentConfig(experiment)

        d = e['with_config']
        self.assertEqual(10, d.intval)
        self.assertEqual(5, d.demo2.intval)
        self.assertTrue(d.experiment_config is e)

        self.assertEqual({'demo2', 'with_config', 'intval'}, e.factories.keys())
        self.assertEqual(5, e.factories['intval'].create())

        d = e.factories['with_config'].create()
        self.assertEqual(10, d.intval)
        self.assertEqual(5, d.demo2.intval)
        self.assertTrue(d.experiment_config is e)

    def test_unordered_nested_config(self):
        experiment = {
            'democ': {
                '_name': 'DemoC'
            },
            'demob': {
                '_name': 'DemoB'
            },
            'demoa': {
                '_name': 'DemoA'
            },
            'simple_int': 2
        }

        e = ExperimentConfig(experiment)
        self.assertEqual(e['demoa'].simple_int, 2)
        self.assertEqual(e['demoa'].attra, None)
        self.assertIsInstance(e['demob'].demoa, DemoA)
        self.assertIsInstance(e['democ'].demob, DemoB)
        self.assertIsInstance(e['democ'].demob.demoa, DemoA)

    def test_nesting_two_levels(self):
        experiment = {
            'democ': {
                '_name': 'DemoC',
                'demob': {
                    '_name': 'DemoB',
                    'attrb': 10,
                    'demoa': {
                        '_name': 'DemoA'
                    }
                }
            },
            'simple_int': 2
        }

        e = ExperimentConfig(experiment)
        self.assertEqual(e['democ'].demob.attrb, 10)
        self.assertEqual(e['democ'].attrc, 3)
        self.assertEqual(e['democ'].demob.demoa.simple_int, 2)

    def test_unsubstituted_param(self):

        experiment = {
            "bar": 'foo',
            "item": {
                "_name": "DemoWithStr",
                "strval": "$bar"
            }
        }
        e = ExperimentConfig(experiment)
        self.assertEqual(e['item'].strval, 'foo')

        experiment = {
            "item": {
                "_name": "DemoWithStr",
                "strval": "$bar"
            }
        }
        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnconfiguredItemsException as e:
            self.assertEqual(len(e.items), 1)
            self.assertEqual({'strval'}, e.items['item'])

        experiment = {
            'demo': {
                '_name': 'DemoWithDict',
                'simple_int': 22,
                'children': {
                    'child0': "$demo3"
                }
            }
        }

        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnconfiguredItemsException as e:
            self.assertEqual(len(e.items), 1)
            self.assertEqual({'$demo3'}, e.items['demo.children.child0'])

        experiment = {
            'demo': {
                '_name': 'DemoWithList',
                'simple_int': 22,
                'children': ["$demo3"]
            }
        }

        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnconfiguredItemsException as e:
            self.assertEqual(len(e.items), 1)
            self.assertEqual({'$demo3'}, e.items['demo.children.0'])

    def test_additional_params(self):

        experiment = {
            "bar": 5,
            "item": {
                "_name": "DemoWithInt",
                "intval": "$bar",
                "bad_param": 2
            }
        }
        try:
            ExperimentConfig(experiment)
            self.fail()
        except BadParameter as b:
            self.assertEqual(b.param, 'bad_param')
            self.assertEqual(b.clazz, 'DemoWithInt')

    def test_bad_plugin(self):

        experiment = {
            "item": {
                "_name": "NoConfig",
            }
        }
        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnknownPluginException as e:
            self.assertEqual(e.clazz, 'NoConfig')

        experiment = {
            "item": {
                "_name": "DemoWithDict",
                'children': {
                    'child': {
                        "_name": "NoConfig"
                    }
                }
            }
        }
        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnknownPluginException as e:
            self.assertEqual(e.clazz, 'NoConfig')

        experiment = {
            "item": {
                "_name": "DemoWithList",
                'children': [
                    {
                        "_name": "NoConfig"
                    }
                ]
            }
        }
        try:
            ExperimentConfig(experiment)
            self.fail()
        except UnknownPluginException as e:
            self.assertEqual(e.clazz, 'NoConfig')

    def test_recursive_list(self):
        experiment = {
            'demo': {
                '_name': 'DemoWithList',
                'simple_int': 22,
                'children': [
                    {
                        '_name': 'DemoWithStr',
                        'strval': 'foo'
                    },
                    '$demo3']
            },
            'demo3': {
                '_name': 'DemoWithInt',
                'intval': 2
            }
        }
        e = ExperimentConfig(experiment)
        demo: DemoWithList = e.experiment['demo']
        self.assertEqual(22, demo.simple_int)

        self.assertIsInstance(demo.children[0], DemoWithStr)
        self.assertEqual(demo.children[0].strval, 'foo')

        self.assertIsInstance(demo.children[1], DemoWithInt)
        self.assertEqual(demo.children[1].intval, 2)

        self.assertEqual(e.factories.keys(), {'demo', 'demo3', 'demo.children', 'demo.children.0', 'demo.children.1'})

        copy = e.factories['demo.children'].create()
        self.assertIsInstance(copy[0], DemoWithStr)
        self.assertEqual(copy[0].strval, 'foo')

        self.assertIsInstance(copy[1], DemoWithInt)
        self.assertEqual(copy[1].intval, 2)

        copy = e.factories['demo.children.0'].create()
        self.assertIsInstance(copy, DemoWithStr)
        self.assertEqual(copy.strval, 'foo')

        copy = e.factories['demo.children.1'].create()
        self.assertIsInstance(copy, DemoWithInt)
        self.assertEqual(copy.intval, 2)

    def test_recursive_dict(self):
        experiment = {
            'demo': {
                '_name': 'DemoWithDict',
                'simple_int': 22,
                'children': {
                    'child0': {
                        '_name': 'DemoWithStr',
                        'strval': 'foo'
                    },
                    'child1': "$demo3"
                }
            },
            'demo3': {
                '_name': 'DemoWithInt',
                'intval': 2
            }
        }

        e = ExperimentConfig(experiment)
        demo: DemoWithDict = e.experiment['demo']
        self.assertEqual(22, demo.simple_int)

        self.assertIsInstance(demo.children['child0'], DemoWithStr)
        self.assertEqual(demo.children['child0'].strval, 'foo')

        self.assertIsInstance(demo.children['child1'], DemoWithInt)
        self.assertEqual(demo.children['child1'].intval, 2)

        self.assertEqual(e.factories.keys(), {'demo', 'demo3', 'demo.children', 'demo.children.child0', 'demo.children.child1'})

        copy = e.factories['demo.children'].create()
        self.assertIsInstance(copy['child0'], DemoWithStr)
        self.assertEqual(copy['child0'].strval, 'foo')

        self.assertIsInstance(copy['child1'], DemoWithInt)
        self.assertEqual(copy['child1'].intval, 2)

        copy = e.factories['demo.children.child0'].create()
        self.assertIsInstance(copy, DemoWithStr)
        self.assertEqual(copy.strval, 'foo')

        copy = e.factories['demo.children.child1'].create()
        self.assertIsInstance(copy, DemoWithInt)
        self.assertEqual(copy.intval, 2)

    def test_method_config(self):
        experiment = {
            'demo': {
                '_name': 'DemoWithStr',
                'strval': "foo",
            },
            "object_from_method": {
                "_name": "demo_method_with_str",
                "str_val": 5
            }
        }

        # Test that the initialization is correct
        e = ExperimentConfig(experiment)
        self.assertIsInstance(e['object_from_method'], DemoWithStr)
        self.assertEqual(e['object_from_method'].strval, 5)

        # Test that we can reconfigure the object from the factory
        object_from_method = e.factories['object_from_method'].create()
        self.assertIsInstance(object_from_method, DemoWithStr)
        self.assertEqual(object_from_method.strval, 5)

    def test_nested_lists_dicts(self):

        experiment = {
            'pipeline': {
                '_name': 'Pipeline',
                'steps': [
                    [['first', '$first'], '$first'],
                    ['second', '$second'],
                ]
            },
            'first': {
                '_name': 'DemoWithInt',
                "intval": 2
            },
            'second': {
                '_name': 'DemoWithInt',
                "intval": 1
            },
            'pipeline_list_of_dict_objects': {
                "_name": "Pipeline",
                "steps": [{
                    '_name': 'DemoWithInt',
                    "intval": 10
                },
                    {
                        '_name': 'DemoWithInt',
                        "intval": 20
                    },
                    {
                        "k1": "v1",
                        "k2": {
                            "_name": "DemoWithInt",
                            "intval": 0
                        },
                        "k3": ['second', '$second']},
                    {
                        "k1": 1,
                        "k2": 2}
                ]
            }
        }
        e = ExperimentConfig(experiment)

        self.assertEqual(e['pipeline'].steps[0][0][0], 'first')
        self.assertIsInstance(e['pipeline'].steps[0][0][1], DemoWithInt)
        self.assertIsInstance(e['pipeline'].steps[0][1], DemoWithInt)
        self.assertIsInstance(e['pipeline'].steps[1][1], DemoWithInt)
        self.assertEqual(e['pipeline'].steps[1][0], 'second')

        # # Test tha factory creation works as expected
        a = e.factories['pipeline.steps.0.0.1'].create()
        self.assertIsInstance(a, DemoWithInt)
        a = e.factories['pipeline.steps.0.1'].create()
        self.assertIsInstance(a, DemoWithInt)
        a = e.factories['pipeline.steps.1.1'].create()
        self.assertIsInstance(a, DemoWithInt)

        a = e.factories['pipeline.steps.1'].create()
        self.assertIsInstance(a, list)
        self.assertIsInstance(a[1], DemoWithInt)
        self.assertEqual(a[0], 'second')

        a = e.factories['pipeline.steps'].create()
        self.assertIsInstance(a, list)
        self.assertIsInstance(a[1], list)
        self.assertIsInstance(a[0], list)
        self.assertIsInstance(a[1][1], DemoWithInt)
        self.assertIsInstance(a[0][0][1], DemoWithInt)

        self.assertIsInstance(e['pipeline_list_of_dict_objects'].steps[0], DemoWithInt)
        self.assertIsInstance(e['pipeline_list_of_dict_objects'].steps[1], DemoWithInt)
        self.assertIsInstance(e['pipeline_list_of_dict_objects'].steps[2]['k2'], DemoWithInt)
        self.assertEqual(e['pipeline_list_of_dict_objects'].steps[3], {
            "k1": 1,
            "k2": 2})

        a = e.factories['pipeline_list_of_dict_objects.steps'].create()
        self.assertIsInstance(a, list)
        a = e.factories['pipeline_list_of_dict_objects.steps.0'].create()
        self.assertIsInstance(a, DemoWithInt)
        a = e.factories['pipeline_list_of_dict_objects.steps.2.k2'].create()
        self.assertIsInstance(a, DemoWithInt)
        a = e.factories['pipeline_list_of_dict_objects.steps.2.k3.1'].create()
        self.assertIsInstance(a, DemoWithInt)
