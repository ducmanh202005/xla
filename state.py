"""Small explicit state transitions; widgets never trigger GPU calls."""


def pair_changed(state, pair_id):
    if state.get('pair_id') == pair_id:
        return
    state.update(pair_id=pair_id, full_result=None, roi_result=None, rois={},
                 last_error=None, selection_revision=state.get('selection_revision', 0) + 1)


def roi_changed(state, side, roi):
    value = list(roi) if roi is not None else None
    rois = dict(state.get('rois', {}))
    if rois.get(side) != value:
        rois[side] = value
        state.update(rois=rois, roi_result=None, last_error=None)
