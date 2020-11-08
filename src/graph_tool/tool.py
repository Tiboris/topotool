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
    ctx = load_context(ctx, storage)



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
@click.option("-m", "--master", default="A")
@click.pass_context
def playbook(ctx, master, storage, output):
    ctx = load_context(ctx, storage)

    G = ctx.obj[GRAPH_NX]

    actual = master
    levels = [[actual]]
    processed = []

    to_proccess = [actual]

    while True:
        if not to_proccess:
            break

        new = []
        for process in to_proccess:
            # print(process)
            # print(type(process))
            # print(G.adj[process])
            for succ in list(G.adj[process]):
                if succ in processed or succ in to_proccess:
                    continue
                # print("adding", succ)
                if succ not in new:
                    # data_4.in demostrates this if node H
                    # is at last level accessed by two predecesors
                    new.append(succ)

        processed = processed + to_proccess
        to_proccess = []
        # print("cleared", to_proccess)
        to_proccess = new
        # print("added", to_proccess)
        if new:
            levels = levels + [new]

    processed_all = sorted(G.nodes) == sorted(processed)
    print("Processed all:", processed_all)
    if not processed_all:
        all_nodes = set(sorted(G.nodes))
        processed = set(sorted(processed))
        print(f"All_nodes:{all_nodes}\nProcessed:{processed}")
        print(
            "Differs in the node(s):",
            list(all_nodes.symmetric_difference(processed)),
        )

    for level in levels:
        print(level)


@graphcli.command()
@click.option("-o", "--output", default="./graph_inventory")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def inventory(ctx, output, storage):
    ctx = load_context(ctx, storage)

    print(output)


if __name__ == "__main__":
    sys.exit(graphcli())
