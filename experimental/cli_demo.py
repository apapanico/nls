import click
import yaml


def parse_kwarg_input(s):
    if '-' in s:
        cov_fun, kwargs = s.split('-')
        cov_fun = cov_fun.lower()
        kwargs = yaml.load(kwargs.replace(':', ': '))
        if kwargs is None:
            kwargs = {}
    else:
        cov_fun = s.lower()
        kwargs = {}
    return cov_fun, kwargs


class KwargParamType(click.ParamType):
    name = 'mixed'

    def convert(self, value, param, ctx):
        try:
            return parse_kwarg_input(value)
        except Exception:
            self.fail('%s is not a valid input' % value, param, ctx)


KWARG = KwargParamType()


@click.command()
@click.option('--item', type=KWARG, default=('slr-{}'))
def demo(item):
    print(item, type(item[0]), type(item[1]))


if __name__ == "__main__":
    demo()
