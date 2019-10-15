import torch


def save_all_model(model, filepath):
    """
    save all model
    this case for to be used model by someone else with no access to this code"""
    torch.save(model, filepath)


def load_model(model, filepath):
    """load model for test"""
    model.load_state_dict(torch.load(filepath))
    model.eval()


def save_model(model, filepath):
    """save model for test"""
    torch.save(model.state_dict(), filepath)


def load_checkpoint(state, model, optimizer):
    """load previous state of model"""
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


def save_checkpoint(state, filename):
    """save current state of model"""
    torch.save(state, filename)
