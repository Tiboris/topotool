# topotool

A utility script which aims to resolve issues with FreeIPA replication topologies and the topology deployments.

```
Usage: topotool [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

  topotool - a tool for tweaking and deploying FreeIPA replication topology

Options:
  -s, --storage TEXT  Location of the context db json file  [default:
                      ./.graph_storage.json]

  --help              Show this message and exit.

Commands:
  analyze     Print infromation about loaded topology graph.
  deployment  Generate deployment files for the Jenkins automation using...
  draw        Create a topology graph picture.
  fixup       Generate an Ansible automation to remove weak spots from the...
  generate    Generate a topology graph based on options.
  load        Loads grap data to context dictionary

```

## Istallation of the tool

```
git clone https://github.com/Tiboris/topotool.git
mkdir venv
python3 -m venv .
cd venv
source ./bin/activate
pip3 install -e ..
```

## Example usage

All the following commands can be chained and does not have to be used as separate command e.g.:
```
$ topotool generate --length=3 --branches=5 fixup --add-while-removing
========================================
----------------------------------------
Added edges:
Added edge(s) count: 0
----------------------------------------
Warning: Removal of the ('y0', 'y1x0') edge created articulation point
Info: Trying to fix new articulation points
----------------------------------------
Added edges:
('y1x1', 'y1x0')
Added edge(s) count: 1
----------------------------------------
----------------------------------------
Removed edges:
('y0', 'y1x0')
('y2', 'y1x1')
Removed edge(s) count: 2
----------------------------------------
Summary:
Added edge(s) [1]:
('y1x1', 'y1x0')
Removed edge(s) [2] :
('y0', 'y1x0')
('y2', 'y1x1')
Difference: set()
----------------------------------------
Saving result graph image
----------------------------------------
Saving result graph data
----------------------------------------
Generating fixup playbook
========================================
```

### Loading the topology

We can load the existing topology from the IPA or a custom from file containing the edges.
An option `-m/--master` should be used to mark the first FreeIPA server from the topology.
Default is set to `--master=y0`

Load a existing topology from the ipa topologysegement-find output (e.g [data_example/data_deployment.in](data_example/data_deployment.in)):
```
$ topotool load --master=y0.ipadomain.test ../data_example/data_deployment.in
# or the same using the type
$ topotool load --type=ipa --master=y0.ipadomain.test ../data_example/data_deployment.in
```
Default behaviour is to load the ipa command output structured file, however we can load custom topology from a file.

You can define a topology by only edges and can load it to topotool.
An example file is in [data_example/data_2loops.in](data_example/data2loops.in)
```
$ topotool load --type=edges ../data_example/data_2loops.in
```
The file format should follow the example one and first line (edge list) is optional.
Connection (edge) list should be a node names separated by `|` which is a parser delimiter for the edge.


### Generate the topology

We can generate a topology for the FreeIPA using two method specifying number of branches and length or by specifying number of required nodes to be in the topology.

Generate topology using branches and length options:
```
topotool generate --length=4 --branches=5
```
both options are optional and default is set to `branches=3` and `length=3`

To generate the topology for specific number of nodes please run the command:
```
topotool generate --nodes=6
```

These two approaches are not the same and topology graphs are not equal.

### Running topotool analyze and draw

A prerequisite to run these commands is either to load or generate the topology.

Topology draw command uses a networkx library and mathplotlib to create a figure of the replication agreement toplogy graph.
You can simply run it by following command:
```
$ topotool draw
```
Information about the loaded topology could be printed out using the:
```
$ topotool analyze
```

### Running fixup command

A prerequisite to run this commands is either to load or generate the topology.

The `fixup` command aims to update the topology replication graph so the topology follows the FreeIPA best practices for the replication agreements.
An exaple run for the pre-generated topology with options `--branches=5 --length=3`
```
$ topotool fixup
```
Output is a pictures of the fixed graph topology, step by step pictures, and playbook which can be run on system to apply these changes to the deployed topology.


### Running deployment command

#TBA

### Running fixup