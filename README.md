# constanzafierro.github.io

Personal site, built with [Jekyll](https://jekyllrb.com/) and the
[minima](https://github.com/jekyll/minima) theme. Deployed by GitHub Pages from
`master`.

## Run locally

Copy-paste from the repo root:

```bash
export PATH="$HOME/.rubies/ruby-3.4.1/bin:$HOME/.gem/ruby/3.4.1/bin:$PATH"
export GEM_HOME="$HOME/.gem/ruby/3.4.1"
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

The two `export` lines point the shell at the chruby-installed Ruby 3.4.1
instead of macOS's system Ruby 2.6, which is too old for this Jekyll. Add them
to `~/.zshrc` (or run `chruby 3.4.1`) to skip them next time.

`jekyll serve` watches for changes, so edits to pages, posts and `_data/` show
up on refresh. Changes to `_config.yml` need a restart.

## Editing the publications list

Selected publications live in `_data/publications.yml` — that is the only file
to edit to add, remove or reorder them. Order in the file is the order on the
page.

```yaml
- title: "Paper title"
  authors: "<b>Constanza Fierro</b>, Coauthor One, Coauthor Two"
  venue: "Conference"
  year: 2025
  paper: "https://..."
  code: "https://github.com/..."
```

Wrap your own name in `<b>...</b>` to bold it. `paper` and `code` are optional:
drop either line and that link disappears. If the file has no entries, the
whole section is omitted.

Rendering lives in `_includes/publications.html`, included from
`_layouts/about.html`; styles are at the bottom of `assets/main.scss`.

## Layout

- `index.md` — the about page (`about` layout).
- `_data/publications.yml` — selected publications.
- `_posts/` — blog posts. Still built and reachable by URL, but deliberately
  not linked from anywhere in the site.
- `assets/main.scss` — all custom styles, on top of minima.
