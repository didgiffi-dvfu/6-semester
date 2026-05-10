function publishPost() {
    const title = document.getElementById('newPostTitle').value;
    const text = document.getElementById('newPostContent').value;

    if (title.trim() === '' || text.trim() === '') {
        alert('Заполни все поля ✨');
        return;
    }

    alert(`Пост "${title}" опубликован ✨`);

    document.getElementById('newPostTitle').value = '';
    document.getElementById('newPostContent').value = '';
}

const likeButtons = document.querySelectorAll('.like-btn');

likeButtons.forEach((button) => {
    button.addEventListener('click', () => {
        button.innerHTML = '❤ Добавлено';
    });
});

const saveButtons = document.querySelectorAll('.save-btn');

saveButtons.forEach((button) => {
    button.addEventListener('click', () => {
        alert('Сохранено ✨');
    });
});
