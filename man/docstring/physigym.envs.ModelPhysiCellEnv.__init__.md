# physigym.envs.ModelPhysiCellEnv.__init__()


## input:
```
    settingxml: string; default is "config/PhysiCell_settings.xml"
        path and filename to the settings.xml file.
        the file will be loaded with lxml and stored at self.x_root.
        therefor all data from the setting.xml file is later on accessible
        via the self.x_root.xpath("//xpath/string/") xpath construct.
        study this source code class for explicit examples.
        for more information about xpath study the following links:
        + https://en.wikipedia.org/wiki/XPath
        + https://www.w3schools.com/xml/xpath_intro.asp
        + https://lxml.de/xpathxslt.html#xpat

    cell_type_cmap: dictionary of strings or string; default viridis.
        dictionary that maps labels to colors strings.
        matplotlib colormap string.
        https://matplotlib.org/stable/tutorials/colors/colormaps.html

    figsize: tuple of floats; default is (8, 6) which is a 4:3 ratio.
        values are in inches (width, height).

    render_mode: string as specified in the metadata or None; default is None.

    render_fps: float or None; default is 10.
        if render_mode is "human", for every dt_gym step the image,
        specified in the physigym.ModelPhysiCellEnv.get_img() function,
        will be generated and displayed. this frame per second setting
        specifies the time the computer sleeps after the image is
        displayed.
        for example 10[fps] = 1/10[spf] = 0.1 [spf].

    verbose: boolean
        to set standard output verbosity true or false.
        please note, only little from the standard output is coming
        actually from physigym. most of the output comes straight
        from PhysiCell and this setting has no influence over that output.

    **kwargs:
        possible additional keyword arguments input.
        will be available in the instance through self.kwargs["key"].

```

## output:
```
    initialized PhysiCell Gymnasium environment.

```

## run:
```python
    import gymnasium
    import physigym

    env = gymnasium.make("physigym/ModelPhysiCellEnv")

    env = gymnasium.make(
        "physigym/ModelPhysiCellEnv",
        settingxml = "config/PhysiCell_settings.xml",
        figsize = (8, 6),
        render_mode = None,
        render_fps = 10,
        verbose = True
    )

```

## description:
```
    function to initialize the PhysiCell Gymnasium environment.

```
