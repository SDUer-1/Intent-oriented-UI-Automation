import torch
import argparse
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW, get_linear_schedule_with_warmup
import transformers
transformers.logging.set_verbosity_error()
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from Dataloader import CustomizedSequences
from train import train, resume
from model import BertNSPM



# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_id', type=str, default='BSAP_1')
    parser.add_argument('--save_exp_id', type=str, default='BSAP_4')

    parser.add_argument('--output_dir', type=str, default='./model_save')

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--train', type=int, default=1, help='train the model')
    parser.add_argument('--train_mode', type=str, default='pretrain')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # custom dataset
    if args.train_mode == 'finetune':
      all_pairs_file_path = '../all_pairs_no_duplication_include_only_task.json'
    if args.train_mode == 'pretrain':
      all_pairs_file_path = '../all_pairs_no_duplication.json'
    
    print("Loading Datasets...")
    sequences_dataset = CustomizedSequences(all_pairs_file_path, tokenizer, args.train_mode)
    print("Successfully Load Datasets...")
    train_size = int(0.75 * len(sequences_dataset))
    val_size = len(sequences_dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(sequences_dataset, [train_size, val_size])

    # dataloaders
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        num_workers=32,
        pin_memory=True
    )
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
        num_workers=32,
        pin_memory=True
    )

    dataloaders = (train_dataloader, validation_dataloader)
    
    if args.resume == 1:
        # resume the trained model
        bert_nspm, tokenizer = resume(args)
        bert_nspm.to(device)
    else:
        # Load pretrained BertModel
        bert_base = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        bert_nspm = BertNSPM(bert_base).to(device)
    
    optimizer = AdamW(bert_nspm.parameters(),
                      lr=args.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    total_steps = len(train_dataloader) * args.epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    if args.train == 1:  # train mode
        train(args, bert_nspm, optimizer, scheduler, tokenizer, dataloaders)
