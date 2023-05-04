import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename, mode=''):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='40', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='40', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')

    g.body.extend(['rankdir=LR'])

    # g.body.extend(['ratio=0.15'])
    # g.view()

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)

            if mode == 'cue' and op != 'skip_connect' and op != 'noise':
                g.edge(u, v, label=op, fillcolor='gray', color='red', fontcolor='red')
            else:
                g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
        # print(genotype)
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    mode = 'cue'
    path = '../../figs/genotypes/cnn_{}/'.format(mode)
    # print(genotype.normal)
    plot(genotype.normal, path + genotype_name + "_normal", mode=mode)
    plot(genotype.reduce, path + genotype_name + "_reduce", mode=mode)
