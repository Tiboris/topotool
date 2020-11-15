#!/usr/bin/env python3
import os
import click
import sys
import shelve

import networkx as nx
import matplotlib.pyplot as plt

from copy import deepcopy
from graph_tool.graphs import Graph


GRAPH_DATA = "graph_input_data"
GRAPH_OBJECT = "graph_object"
GRAPH_NX = "graph_networkx"
LEVELS = "levels"
PRED = "predecessors"
BACKBONE = "backbone"
MASTER = "master_node"


def load_context(ctx, storage):
    ctx.ensure_object(dict)
    if not ctx.obj:
        try:
            with shelve.open(storage) as db:
                ctx.obj[GRAPH_DATA] = db[GRAPH_DATA]
                ctx.obj[GRAPH_OBJECT] = db[GRAPH_OBJECT]
                ctx.obj[GRAPH_NX] = db[GRAPH_NX]
                ctx.obj[LEVELS] = db[LEVELS]
                ctx.obj[PRED] = db[PRED]
                ctx.obj[MASTER] = db[MASTER]
        except KeyError:
            pass

    return ctx


@click.group(chain=True)
@click.option("--debug", default=False, is_flag=True)
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.option("-o", "--output", default="./graph_out")
@click.pass_context
def graphcli(ctx, storage, output, debug):
    """graphcli tool for topology"""
    ctx = load_context(ctx, storage)


def levels_from_fist_successors(successors):
    return [[successors[0][0]], list(successors[0][1])]


def predecessors_from_first_levels(levels):
    res = {}
    for node in levels[-1]:
        res[node] = levels[0][0]
    # print(res)
    return res


@graphcli.command()
@click.argument("input_file")
@click.option("-d", "--delimiter", default="-")
@click.option("-w", "--weight-delimiter", default="")
@click.option("-m", "--master", default="y0")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def load(ctx, input_file, delimiter, weight_delimiter, master, storage):
    """Loads grap data to dictionary"""
    ctx.ensure_object(dict)

    with open(input_file, "r") as f:
        ctx.obj[GRAPH_DATA] = f.read().splitlines()

    data = deepcopy(ctx.obj[GRAPH_DATA])
    graph_data = Graph(
        data, delimiter=delimiter, weight_delim=weight_delimiter
    )
    ctx.obj[GRAPH_OBJECT] = graph_data

    G = nx.Graph()
    for vertex in graph_data.vertices:
        G.add_node(vertex)

    G.add_edges_from(graph_data.edge_list)
    ctx.obj[GRAPH_NX] = G

    all_successors = list(nx.bfs_successors(G, master))

    levels = levels_from_fist_successors(all_successors[:1])
    successors = all_successors[1:]

    new = []
    predecessors = predecessors_from_first_levels(levels)

    for s, f in successors:
        # print("------------------------------------")
        # print("s:", s, "f:", f, "level:", levels[-1])
        if s in levels[-1]:
            # print(s, f)
            new = new + f
            # print("+")
        else:
            # print(s, f)
            # print("next", f, "level:", new)
            # print("before", levels)
            levels.append(new)
            # print("afeter", levels)
            new = f
        for pred in f:
            predecessors[pred] = s

    if new:  # add level at the end of for cycle so last is added
        levels.append(new)

    key = 0
    level_dict = {}
    for level in levels:
        level_dict[key] = level
        # print(level)
        key += 1

    # for level, replicas in level_dict.items():  # for inventory
    #     print("---")
    #     print(level, replicas)
    #     if level:
    #         for replica in replicas:
    #             print(replica, predecessors[replica])

    ctx.obj[LEVELS] = level_dict
    ctx.obj[PRED] = predecessors
    ctx.obj[MASTER] = master

    with shelve.open(storage) as db:
        for key in ctx.obj:
            db[key] = ctx.obj[key]


@graphcli.command()
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def draw(ctx, storage):
    """Draw a topology graph as picture."""
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-t", "--template", default="../data/inventory_template.j2")  # FIXME
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def inventory(ctx, storage, out_dir, template, output):
    ctx = load_context(ctx, storage)

    try:
        pred = ctx.obj[PRED]
        levels = ctx.obj[LEVELS]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    print(levels, pred)


@graphcli.command()
@click.option("-o", "--output", default="./graph_inventory")
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-t", "--template", default="../data/jenkinsjob.j2")  # FIXME
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def jenkinsfile(ctx, output, out_dir, storage):
    ctx = load_context(ctx, storage)

    try:
        levels = ctx.obj[LEVELS]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    print(levels)


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-t", "--template", default="../data/inventory_template.j2")  # FIXME
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def playbook(ctx, storage, out_dir, template, output):
    ctx = load_context(ctx, storage)

    try:
        pred = ctx.obj[PRED]
        levels = ctx.obj[LEVELS]
        G = ctx.obj[GRAPH_NX]
        master = ctx.obj[MASTER]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    print(levels, pred)

    backbone = list(nx.bfs_edges(G, master))

    print("BACKBONE", backbone)

    print(sorted(G.edges))
    print(sorted(backbone))
    ctx.obj[BACKBONE] = backbone






if __name__ == "__main__":
    sys.exit(graphcli())
