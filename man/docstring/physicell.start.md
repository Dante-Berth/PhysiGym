# physicell.start()

## input:
```
    settingxml 'path/to/setting.xml' file (string); default is 'config/PhysiCell_settings.xml'.
    reload (bool) density and parameter structs; default is False.

```

## output:
```
    PhysiCell processing. 0 for success.

```

## run:
```python
    from extending import physicell
    physicell.start('path/to/setting.xml')

```

## description:
```
    function (re)initializes PhysiCell as specified in the settings.xml, cells.csv, and cell_rules.csv files and generates the step zero observation output.
```