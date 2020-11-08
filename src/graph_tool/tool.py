#!/usr/bin/env python3
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


def load_context(ctx, storage):
    if not ctx.obj:
        ctx.ensure_object(dict)
        with shelve.open(storage) as db:
            ctx.obj[GRAPH_DATA] = db[GRAPH_DATA]
            ctx.obj[GRAPH_OBJECT] = db[GRAPH_OBJECT]
            ctx.obj[GRAPH_NX] = db[GRAPH_NX]

    return ctx


@click.group(chain=True)
@click.option("--debug", default=False, is_flag=True)
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.option("-o", "--output", default="./graph_out")
@click.pass_context
def graphcli(ctx, storage, output, debug):
    """graphcli tool for topology"""
    ctx.ensure_object(dict)
    with shelve.open(storage) as db:
        db["data"] = ctx.obj


@graphcli.command()
@click.argument("input_file")
@click.option("-d", "--delimiter", default="-")
@click.option("-w", "--weight-delimiter", default="")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def load(ctx, input_file, delimiter, weight_delimiter, storage):
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

    with shelve.open(storage) as db:
        for key in ctx.obj:
            db[key] = ctx.obj[key]


@graphcli.command()
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def draw(ctx, storage):
    """Draw a topology graph as picture."""
    ctx = load_context(ctx, storage)

    G = ctx.obj[GRAPH_NX]
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def playbook(ctx, storage, output):
    ctx = load_context(ctx, storage)

    G = ctx.obj[GRAPH_NX]

    levels = [["A"], list(G.adj["A"])]

    print(levels)


@graphcli.command()
@click.option("-o", "--output", default="./graph_inventory")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def inventory(ctx, output, storage):
    ctx = load_context(ctx, storage)

    print(output)


if __name__ == "__main__":
    sys.exit(graphcli())
